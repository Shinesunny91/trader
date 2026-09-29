package com.nseintradayai.app;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.job.JobInfo;
import android.app.job.JobParameters;
import android.app.job.JobScheduler;
import android.app.job.JobService;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.net.Uri;
import android.os.Build;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.HashSet;
import java.util.Set;

/**
 * Polls the workspace's ticket API and raises a notification when a genuinely
 * new trade appears.
 *
 * <p>Uses the framework {@link JobScheduler} rather than WorkManager because
 * this APK is built without any dependency graph — aapt2 and ecj only, no
 * Gradle resolution — and JobScheduler has been in the framework since API 21.
 * Android caps periodic jobs at one run per 15 minutes, which is the right
 * cadence anyway: the book publishes on a 5-minute timer and a phone that
 * buzzes more often than a trade actually arrives gets muted by its owner.
 *
 * <p>Dedup matters more than latency here. The API returns a stable
 * {@code ticket_ids} array, and already-seen ids are kept in preferences, so a
 * ticket that stays live across several polls notifies exactly once.
 */
public class TicketJobService extends JobService {

    static final int JOB_ID = 4101;
    static final String CHANNEL_ID = "trade_signals";
    static final String KEY_NOTIFY = "notifications_enabled";
    static final String KEY_SEEN = "seen_ticket_ids";
    private static final int NOTIFICATION_ID = 7301;

    @Override
    public boolean onStartJob(final JobParameters params) {
        // Anonymous Runnable rather than a lambda: this APK is compiled by ecj
        // against android.jar alone, which has no java.lang.invoke.
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    poll();
                } catch (Exception ignored) {
                    // A failed poll is normal: the laptop is off, or the phone
                    // left the Wi-Fi. Reschedule silently rather than nagging.
                } finally {
                    jobFinished(params, false);
                }
            }
        }).start();
        return true;    // work continues on the background thread
    }

    @Override
    public boolean onStopJob(JobParameters params) {
        return true;    // ask to be rescheduled
    }

    private void poll() throws Exception {
        SharedPreferences prefs = getSharedPreferences(Storage.PREFS, Context.MODE_PRIVATE);
        if (!prefs.getBoolean(KEY_NOTIFY, true)) {
            return;
        }
        String base = prefs.getString(MainActivity.SERVER_URL_KEY, MainActivity.DEFAULT_SERVER_URL);
        String body = fetch(apiUrl(base, "/tickets"));
        if (body == null) {
            return;
        }

        JSONObject payload = new JSONObject(body);
        // Keep an offline copy on the chosen volume — the phone stays useful
        // when the laptop is off.
        Storage.write(this, "today_tickets.json", body);

        JSONArray ids = payload.optJSONArray("ticket_ids");
        if (ids == null || ids.length() == 0) {
            return;
        }
        Set<String> seen = new HashSet<>(prefs.getStringSet(KEY_SEEN, new HashSet<>()));
        JSONArray tickets = payload.optJSONArray("tickets");
        String fresh = null;
        int freshCount = 0;
        for (int i = 0; i < ids.length(); i++) {
            String id = ids.optString(i);
            if (id.isEmpty() || seen.contains(id)) {
                continue;
            }
            seen.add(id);
            freshCount++;
            if (fresh == null && tickets != null && i < tickets.length()) {
                JSONObject t = tickets.optJSONObject(i);
                if (t != null) {
                    fresh = t.optString("symbol") + "  " + t.optString("side")
                            + "  qty " + t.optInt("quantity");
                }
            }
        }
        if (freshCount == 0) {
            return;
        }
        // Bound the dedup set so it cannot grow without limit across sessions.
        if (seen.size() > 400) {
            seen = new HashSet<>();
        }
        prefs.edit().putStringSet(KEY_SEEN, seen).apply();
        Storage.append(this, "ticket_history.log",
                payload.optString("generated_at") + "  " + fresh);

        String title = freshCount == 1 ? "New trade ticket" : freshCount + " new trade tickets";
        notify(title, fresh == null ? payload.optString("status") : fresh);
    }

    /** The JSON API runs beside Streamlit; same host, ticket-API port. */
    static String apiUrl(String serverUrl, String path) {
        try {
            URL parsed = new URL(serverUrl);
            int port = MainActivity.API_PORT;
            return new URL(parsed.getProtocol(), parsed.getHost(), port, path).toString();
        } catch (Exception e) {
            return serverUrl + path;
        }
    }

    private String fetch(String url) throws Exception {
        HttpURLConnection conn = (HttpURLConnection) new URL(url).openConnection();
        conn.setConnectTimeout(6000);
        conn.setReadTimeout(6000);
        conn.setRequestProperty("Accept", "application/json");
        try {
            if (conn.getResponseCode() != 200) {
                return null;
            }
            StringBuilder sb = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    sb.append(line);
                }
            }
            return sb.toString();
        } finally {
            conn.disconnect();
        }
    }

    private void notify(String title, String text) {
        NotificationManager manager = getSystemService(NotificationManager.class);
        if (manager == null) {
            return;
        }
        ensureChannel(this);
        Intent open = new Intent(this, MainActivity.class);
        open.setFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP);
        int flags = PendingIntent.FLAG_UPDATE_CURRENT;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            flags |= PendingIntent.FLAG_IMMUTABLE;      // required from API 31
        }
        PendingIntent intent = PendingIntent.getActivity(this, 0, open, flags);

        Notification.Builder builder = Build.VERSION.SDK_INT >= Build.VERSION_CODES.O
                ? new Notification.Builder(this, CHANNEL_ID)
                : new Notification.Builder(this);
        Notification notification = builder
                .setSmallIcon(android.R.drawable.stat_notify_sync)
                .setContentTitle(title)
                .setContentText(text)
                .setStyle(new Notification.BigTextStyle().bigText(text))
                .setContentIntent(intent)
                .setAutoCancel(true)
                .build();
        manager.notify(NOTIFICATION_ID, notification);
    }

    static void ensureChannel(Context context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) {
            return;
        }
        NotificationManager manager = context.getSystemService(NotificationManager.class);
        if (manager == null || manager.getNotificationChannel(CHANNEL_ID) != null) {
            return;
        }
        NotificationChannel channel = new NotificationChannel(
                CHANNEL_ID, "Trade signals", NotificationManager.IMPORTANCE_DEFAULT);
        channel.setDescription("Fires when the paper book publishes a new order ticket.");
        manager.createNotificationChannel(channel);
    }

    /** (Re)schedule the periodic poll. Android floors the period at 15 minutes. */
    static void schedule(Context context, boolean enabled) {
        JobScheduler scheduler = context.getSystemService(JobScheduler.class);
        if (scheduler == null) {
            return;
        }
        scheduler.cancel(JOB_ID);
        if (!enabled) {
            return;
        }
        JobInfo job = new JobInfo.Builder(JOB_ID, new ComponentName(context, TicketJobService.class))
                .setRequiredNetworkType(JobInfo.NETWORK_TYPE_ANY)
                .setPeriodic(15 * 60 * 1000L)
                .setPersisted(true)             // survive reboot
                .build();
        scheduler.schedule(job);
    }
}
