package com.nseintradayai.app;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.graphics.Typeface;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.Gravity;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/**
 * Standalone picks — no laptop, no workspace on the network.
 *
 * <p>Fetches daily bars straight from Yahoo, runs the rule on-device and shows
 * the sized trade. Results are cached to the chosen storage volume (SD card by
 * default), so the last pick survives going offline entirely, and a re-open on
 * the same session costs no network at all: daily bars change once a day.
 *
 * <p>The measured expectancy is printed above the table, not buried in a help
 * screen, because two of the three books lost money over the ten-year test and
 * a picker that hides that is not a research tool.
 */
public class PicksActivity extends Activity {

    private static final String CACHE_FILE = "standalone_picks.json";

    private Spinner bookSpinner;
    private EditText capitalInput;
    private TextView expectancyText;
    private TextView statusText;
    private ProgressBar progress;
    private LinearLayout resultsBox;
    private Button refreshButton;

    private final ExecutorService pool = Executors.newFixedThreadPool(8);
    private final Handler ui = new Handler(Looper.getMainLooper());

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(buildLayout());
        showCached();
    }

    @Override
    protected void onDestroy() {
        pool.shutdownNow();
        super.onDestroy();
    }

    private View buildLayout() {
        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.WHITE);
        root.setPadding(dp(16), dp(14), dp(16), dp(24));

        TextView title = new TextView(this);
        title.setText("Standalone picks");
        title.setTextColor(Color.rgb(15, 23, 42));
        title.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        title.setTextSize(20);
        root.addView(title);

        TextView sub = new TextView(this);
        sub.setText("Runs entirely on this phone. No laptop needed.");
        sub.setTextColor(Color.rgb(100, 116, 139));
        sub.setTextSize(12);
        sub.setPadding(0, dp(2), 0, dp(12));
        root.addView(sub);

        bookSpinner = new Spinner(this);
        List<String> labels = new ArrayList<>();
        for (SwingEngine.Book b : SwingEngine.Book.values()) {
            labels.add(b.label + (b.recommended ? "  ✓ measured positive" : ""));
        }
        ArrayAdapter<String> adapter = new ArrayAdapter<>(
                this, android.R.layout.simple_spinner_item, labels);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        bookSpinner.setAdapter(adapter);
        bookSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int pos, long id) {
                showExpectancy(SwingEngine.Book.values()[pos]);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });
        root.addView(bookSpinner);

        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(0, dp(8), 0, dp(8));

        TextView capLabel = new TextView(this);
        capLabel.setText("Capital ₹");
        capLabel.setTextSize(13);
        capLabel.setTextColor(Color.rgb(51, 65, 85));
        row.addView(capLabel);

        capitalInput = new EditText(this);
        capitalInput.setSingleLine(true);
        capitalInput.setTextSize(14);
        capitalInput.setInputType(android.text.InputType.TYPE_CLASS_NUMBER);
        capitalInput.setText(String.valueOf(
                getSharedPreferences(Storage.PREFS, MODE_PRIVATE)
                        .getLong("swing_capital", 1000000L)));
        row.addView(capitalInput, new LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));

        refreshButton = new Button(this);
        refreshButton.setText("Refresh");
        refreshButton.setAllCaps(false);
        refreshButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                run();
            }
        });
        row.addView(refreshButton);
        root.addView(row);

        expectancyText = new TextView(this);
        expectancyText.setTextSize(12);
        expectancyText.setPadding(dp(10), dp(8), dp(10), dp(8));
        root.addView(expectancyText);

        progress = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progress.setMax(100);
        progress.setVisibility(View.GONE);
        root.addView(progress);

        statusText = new TextView(this);
        statusText.setTextSize(12);
        statusText.setTextColor(Color.rgb(71, 85, 105));
        statusText.setPadding(0, dp(6), 0, dp(6));
        root.addView(statusText);

        resultsBox = new LinearLayout(this);
        resultsBox.setOrientation(LinearLayout.VERTICAL);
        root.addView(resultsBox);

        TextView footer = new TextView(this);
        footer.setText("Entries fill at the NEXT session's open. The stop is a distance "
                + "from that fill, not from the reference close shown here. "
                + "Research tool — not financial advice.");
        footer.setTextSize(11);
        footer.setTextColor(Color.rgb(100, 116, 139));
        footer.setPadding(0, dp(16), 0, 0);
        root.addView(footer);

        showExpectancy(SwingEngine.Book.values()[0]);
        scroll.addView(root);
        return scroll;
    }

    private void showExpectancy(SwingEngine.Book book) {
        expectancyText.setText("Measured on 10 years of daily bars:\n" + book.expectancy);
        expectancyText.setBackgroundColor(book.recommended
                ? Color.rgb(228, 239, 232) : Color.rgb(246, 231, 228));
        expectancyText.setTextColor(book.recommended
                ? Color.rgb(44, 106, 69) : Color.rgb(168, 53, 42));
    }

    private SwingEngine.Book selectedBook() {
        return SwingEngine.Book.values()[bookSpinner.getSelectedItemPosition()];
    }

    private double capital() {
        try {
            double v = Double.parseDouble(capitalInput.getText().toString().trim());
            return v > 0 ? v : 1000000.0;
        } catch (NumberFormatException e) {
            return 1000000.0;
        }
    }

    private void run() {
        final SwingEngine.Book book = selectedBook();
        final double capital = capital();
        getSharedPreferences(Storage.PREFS, MODE_PRIVATE).edit()
                .putLong("swing_capital", (long) capital).apply();

        refreshButton.setEnabled(false);
        progress.setVisibility(View.VISIBLE);
        progress.setProgress(0);
        resultsBox.removeAllViews();
        final String[] universe = book.universe();
        statusText.setText("Fetching " + universe.length + " symbols...");

        new Thread(new Runnable() {
            @Override
            public void run() {
                final List<MarketData.Bars> bars = new ArrayList<>();
                List<Future<MarketData.Bars>> futures = new ArrayList<>();
                for (final String symbol : universe) {
                    futures.add(pool.submit(new java.util.concurrent.Callable<MarketData.Bars>() {
                        @Override
                        public MarketData.Bars call() {
                            try {
                                return MarketData.fetchDaily(symbol);
                            } catch (Exception e) {
                                return null;    // one bad symbol must not sink the scan
                            }
                        }
                    }));
                }
                int done = 0;
                for (Future<MarketData.Bars> f : futures) {
                    try {
                        MarketData.Bars b = f.get(45, TimeUnit.SECONDS);
                        if (b != null) {
                            bars.add(b);
                        }
                    } catch (Exception ignored) {
                        // timeout or failure: skip
                    }
                    done++;
                    final int pct = done * 100 / Math.max(universe.length, 1);
                    ui.post(new Runnable() {
                        @Override
                        public void run() {
                            progress.setProgress(pct);
                        }
                    });
                }
                final List<SwingEngine.Pick> picks =
                        SwingEngine.rank(bars, book, capital, 2.0, 5);
                final int fetched = bars.size();
                ui.post(new Runnable() {
                    @Override
                    public void run() {
                        progress.setVisibility(View.GONE);
                        refreshButton.setEnabled(true);
                        if (picks.isEmpty()) {
                            statusText.setText("No candidate. Fetched " + fetched + " of "
                                    + universe.length + " symbols — check the connection.");
                            return;
                        }
                        String stamp = new SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US)
                                .format(new Date());
                        statusText.setText(book.label + " · " + fetched + "/" + universe.length
                                + " symbols · " + stamp);
                        render(picks, book);
                        cache(picks, book, stamp, fetched);
                    }
                });
            }
        }).start();
    }

    private void render(List<SwingEngine.Pick> picks, SwingEngine.Book book) {
        resultsBox.removeAllViews();
        boolean first = true;
        for (SwingEngine.Pick p : picks) {
            LinearLayout card = new LinearLayout(this);
            card.setOrientation(LinearLayout.VERTICAL);
            card.setPadding(dp(12), dp(10), dp(12), dp(10));
            card.setBackgroundColor(first ? Color.rgb(226, 239, 238) : Color.rgb(244, 246, 245));

            TextView head = new TextView(this);
            head.setText((first ? "TOP PICK   " : "") + p.symbol);
            head.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
            head.setTextSize(first ? 17 : 15);
            head.setTextColor(Color.rgb(15, 23, 42));
            card.addView(head);

            TextView body = new TextView(this);
            body.setTypeface(Typeface.MONOSPACE);
            body.setTextSize(12);
            body.setTextColor(Color.rgb(51, 65, 85));
            body.setText(String.format(Locale.US,
                    "ref close   %.2f\nstop        %.2f  (%.2f below fill)\n"
                            + "quantity    %d\nvalue       Rs %,.0f\nrisk        Rs %,.0f"
                            + "\ncost        %.1f bps round trip\nRSI(14)     %.1f\n"
                            + "hold        %d sessions",
                    p.close, p.close - p.stopDistance, p.stopDistance, p.quantity,
                    p.positionValue, p.riskRupees, p.costBps, p.rsi, book.holdSessions()));
            card.addView(body);

            resultsBox.addView(card, marginParams());
            first = false;
        }
    }

    private LinearLayout.LayoutParams marginParams() {
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.bottomMargin = dp(8);
        return lp;
    }

    /** Persist so the last pick survives going fully offline. */
    private void cache(List<SwingEngine.Pick> picks, SwingEngine.Book book,
                       String stamp, int fetched) {
        try {
            JSONObject root = new JSONObject();
            root.put("generated_at", stamp);
            root.put("book", book.name());
            root.put("label", book.label);
            root.put("expectancy", book.expectancy);
            root.put("symbols_fetched", fetched);
            JSONArray arr = new JSONArray();
            for (SwingEngine.Pick p : picks) {
                JSONObject o = new JSONObject();
                o.put("symbol", p.symbol);
                o.put("close", p.close);
                o.put("stop", p.close - p.stopDistance);
                o.put("quantity", p.quantity);
                o.put("position_value", p.positionValue);
                o.put("risk", p.riskRupees);
                o.put("cost_bps", p.costBps);
                o.put("rsi", p.rsi);
                o.put("hold_sessions", book.holdSessions());
                arr.put(o);
            }
            root.put("picks", arr);
            Storage.write(this, CACHE_FILE, root.toString(2));
        } catch (Exception ignored) {
            // caching is a convenience, never a reason to fail the screen
        }
    }

    private void showCached() {
        try {
            File f = new File(Storage.dataDir(this), CACHE_FILE);
            if (!f.exists()) {
                statusText.setText("Tap Refresh to fetch today's bars.");
                return;
            }
            byte[] buf = new byte[(int) f.length()];
            try (java.io.FileInputStream in = new java.io.FileInputStream(f)) {
                //noinspection ResultOfMethodCallIgnored
                in.read(buf);
            }
            JSONObject root = new JSONObject(new String(buf, "UTF-8"));
            JSONArray arr = root.getJSONArray("picks");
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < arr.length(); i++) {
                JSONObject o = arr.getJSONObject(i);
                sb.append(i == 0 ? "TOP PICK   " : "").append(o.getString("symbol"))
                        .append(String.format(Locale.US,
                                "\n  close %.2f  stop %.2f  qty %d  cost %.1f bps\n",
                                o.getDouble("close"), o.getDouble("stop"),
                                o.getInt("quantity"), o.getDouble("cost_bps")));
            }
            TextView cached = new TextView(this);
            cached.setTypeface(Typeface.MONOSPACE);
            cached.setTextSize(12);
            cached.setTextColor(Color.rgb(51, 65, 85));
            cached.setText(sb.toString());
            resultsBox.addView(cached);
            statusText.setText("Cached from " + root.optString("generated_at")
                    + " (" + root.optString("label") + ") — tap Refresh to update.");
        } catch (Exception e) {
            statusText.setText("Tap Refresh to fetch today's bars.");
        }
    }

    private int dp(int value) {
        return (int) (value * getResources().getDisplayMetrics().density + 0.5f);
    }
}
