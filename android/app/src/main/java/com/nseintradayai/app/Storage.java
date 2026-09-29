package com.nseintradayai.app;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Environment;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;

/**
 * Where the app keeps its offline copy of the book.
 *
 * <p>Android 11 enforces scoped storage, so the old approach — asking for
 * WRITE_EXTERNAL_STORAGE and writing to /sdcard — silently does nothing on an
 * API 30 device. The permission is still declarable and still ignored. What
 * <em>does</em> work without any permission at all is
 * {@link Context#getExternalFilesDirs}, which returns one directory per storage
 * volume: index 0 is internal shared storage, and any later entry is a physical
 * removable card. Writing there needs no runtime grant on any Android version
 * the app supports, and the files are visible over MTP.
 *
 * <p>The trade-off, which is worth stating plainly: app-specific directories
 * are removed when the app is uninstalled. That is the price of not demanding
 * broad storage access, and for a cache of daily tickets it is the right side
 * of the trade.
 *
 * <p>Default is the SD card when one is present and mounted, falling back to
 * internal storage otherwise, so a phone with no card still works.
 */
final class Storage {

    static final String PREFS = "signal_lab_prefs";
    static final String KEY_PREFER_SD = "prefer_sd_card";

    private Storage() {
    }

    static boolean preferSdCard(Context context) {
        return prefs(context).getBoolean(KEY_PREFER_SD, true);   // default ON
    }

    static void setPreferSdCard(Context context, boolean value) {
        prefs(context).edit().putBoolean(KEY_PREFER_SD, value).apply();
    }

    private static SharedPreferences prefs(Context context) {
        return context.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
    }

    /** The removable volume's app directory, or null when there is no card. */
    static File sdCardDir(Context context) {
        File[] volumes = context.getExternalFilesDirs(null);
        if (volumes == null) {
            return null;
        }
        for (int i = 1; i < volumes.length; i++) {          // index 0 is internal
            File volume = volumes[i];
            if (volume == null) {
                continue;
            }
            String state = Environment.getExternalStorageState(volume);
            if (Environment.MEDIA_MOUNTED.equals(state)) {
                return volume;
            }
        }
        return null;
    }

    /** Directory the app should actually write to right now. */
    static File dataDir(Context context) {
        if (preferSdCard(context)) {
            File sd = sdCardDir(context);
            if (sd != null) {
                File dir = new File(sd, "book");
                if (dir.exists() || dir.mkdirs()) {
                    return dir;
                }
            }
        }
        File fallback = new File(context.getExternalFilesDir(null), "book");
        if (!fallback.exists()) {
            //noinspection ResultOfMethodCallIgnored
            fallback.mkdirs();
        }
        return fallback;
    }

    /** Human-readable location for the settings screen. */
    static String describe(Context context) {
        File sd = sdCardDir(context);
        File active = dataDir(context);
        if (sd == null) {
            return "Internal storage (no SD card detected)\n" + active.getAbsolutePath();
        }
        boolean onCard = active.getAbsolutePath().startsWith(sd.getAbsolutePath());
        return (onCard ? "SD card" : "Internal storage (SD card available)")
                + "\n" + active.getAbsolutePath();
    }

    static void write(Context context, String filename, String body) throws IOException {
        File target = new File(dataDir(context), filename);
        try (FileOutputStream out = new FileOutputStream(target);
             OutputStreamWriter writer = new OutputStreamWriter(out, StandardCharsets.UTF_8)) {
            writer.write(body);
        }
    }

    /** Appends one line to a dated log so the phone keeps its own history. */
    static void append(Context context, String filename, String line) throws IOException {
        File target = new File(dataDir(context), filename);
        try (FileOutputStream out = new FileOutputStream(target, true);
             OutputStreamWriter writer = new OutputStreamWriter(out, StandardCharsets.UTF_8)) {
            writer.write(line);
            writer.write("\n");
        }
    }
}
