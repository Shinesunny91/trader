package com.nseintradayai.app;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * The one rule that survived the ten-year test, reimplemented on-device.
 *
 * <p>Measured by {@code scripts/swing_backtest.py} on 10 years of daily bars
 * with Groww costs and risk-based sizing: buying the most oversold commodity
 * future and holding five sessions returned <b>+55.2%</b> over the decade,
 * positive in <b>8 of 10 years</b> and in both halves of the sample, with a
 * 17.6% max drawdown and profit factor 1.34. Its bootstrap confidence interval
 * on bps-per-trade still spans zero, so it is evidence worth paper-trading, not
 * a forecast.
 *
 * <p>NSE runs here too, on the same daily bars and the same arithmetic — the
 * {@code trend_pullback} rule over the 120 most liquid names. What separates
 * the two is evidence, not capability: at a five-session hold NSE measured
 * <b>-14.8%</b> over the decade and every other factor was worse, while a
 * 40-session hold measured <b>+77.3%</b>. Both are offered, both carry their
 * measured number, and {@link Book#recommended} marks the only one that earned
 * it. The difference is the cost floor — ~37 bps a round trip on delivery
 * against ~12 on futures.
 *
 * <p>Deliberately <em>not</em> ported: the 5-minute intraday scanner. That one
 * genuinely cannot run here — it needs a 500-symbol intraday fetch every bar, a
 * trained random forest and the shadow-learner database — and its measured edge
 * is negative regardless. The WebView tab still reaches it when the workspace is
 * on the network.
 *
 * <p>Every number in {@link Pick} is computed from bars that have already
 * closed. The stop is quoted as a distance from the fill, because the fill is
 * the next session's open and is not known yet.
 */
final class SwingEngine {

    static final int RSI_PERIOD = 14;
    static final int ATR_PERIOD = 14;
    static final int MIN_BARS = 220;
    static final double STOP_ATR = 2.5;
    static final int HOLD_SESSIONS = 5;

    /** What the picker is allowed to run, and what it measured. */
    enum Book {
        COMMODITY_5D("Commodities, 5-session hold",
                "+55.2% over 10 years, 8/10 years positive, max drawdown 17.6%, "
                        + "profit factor 1.34, positive in both halves. Bootstrap CI on "
                        + "bps/trade still spans zero.", true),
        NSE_5D("NSE, 5-session hold",
                "-14.8% over 10 years. EVERY factor tested was negative at this "
                        + "hold; NIFTY buy & hold returned +176% over the same window. "
                        + "Shown for completeness — do not trade this for profit.", false),
        NSE_40D("NSE, 40-session hold",
                "+77.3% over 10 years, but that is a two-month position rather "
                        + "than intra-week, and it still lost to NIFTY buy & hold "
                        + "(+176%) over the same decade.", false);

        final String label;
        final String expectancy;
        final boolean recommended;

        Book(String label, String expectancy, boolean recommended) {
            this.label = label;
            this.expectancy = expectancy;
            this.recommended = recommended;
        }

        boolean isCommodity() {
            return this == COMMODITY_5D;
        }

        int holdSessions() {
            return this == NSE_40D ? 40 : 5;
        }

        String[] universe() {
            return isCommodity() ? MarketData.COMMODITIES : MarketData.NSE_LIQUID;
        }
    }

    static final class Pick {
        String symbol;
        double close;
        double rsi;
        double atr;
        double score;
        double stopDistance;
        int quantity;
        double positionValue;
        double riskRupees;
        double costBps;
        String asOf;
    }

    private SwingEngine() {
    }

    /** Wilder's RSI, the same smoothing the Python side uses. */
    static double rsi(double[] close, int period) {
        if (close.length < period + 1) {
            return Double.NaN;
        }
        double gain = 0, loss = 0;
        for (int i = 1; i <= period; i++) {
            double d = close[i] - close[i - 1];
            if (d > 0) {
                gain += d;
            } else {
                loss -= d;
            }
        }
        gain /= period;
        loss /= period;
        double alpha = 1.0 / period;
        for (int i = period + 1; i < close.length; i++) {
            double d = close[i] - close[i - 1];
            gain = (1 - alpha) * gain + alpha * Math.max(d, 0);
            loss = (1 - alpha) * loss + alpha * Math.max(-d, 0);
        }
        if (loss <= 0) {
            return 100.0;
        }
        return 100.0 - 100.0 / (1.0 + gain / loss);
    }

    /** Wilder-smoothed Average True Range. */
    static double atr(double[] high, double[] low, double[] close, int period) {
        int n = close.length;
        if (n < period + 1) {
            return Double.NaN;
        }
        double alpha = 1.0 / period;
        double value = 0;
        for (int i = 1; i <= period; i++) {
            value += trueRange(high, low, close, i);
        }
        value /= period;
        for (int i = period + 1; i < n; i++) {
            value = (1 - alpha) * value + alpha * trueRange(high, low, close, i);
        }
        return value;
    }

    private static double trueRange(double[] high, double[] low, double[] close, int i) {
        double a = high[i] - low[i];
        double b = Math.abs(high[i] - close[i - 1]);
        double c = Math.abs(low[i] - close[i - 1]);
        return Math.max(a, Math.max(b, c));
    }

    /**
     * Risk-based size: a stop-out costs {@code riskPct} of capital regardless of
     * how volatile the contract is, capped so a quiet one cannot become leverage.
     */
    static int positionSize(double entry, double atr, double capital,
                            double riskPct, double maxPositionPct) {
        double perShareRisk = STOP_ATR * atr;
        if (perShareRisk <= 0 || entry <= 0) {
            return 0;
        }
        long byRisk = (long) (capital * riskPct / 100.0 / perShareRisk);
        long byNotional = (long) (capital * maxPositionPct / 100.0 / entry);
        return (int) Math.max(0, Math.min(byRisk, byNotional));
    }

    static double sma(double[] c, int n) {
        if (c.length < n) {
            return Double.NaN;
        }
        double sum = 0;
        for (int i = c.length - n; i < c.length; i++) {
            sum += c[i];
        }
        return sum / n;
    }

    /** Return over the last {@code n} bars, as a fraction. */
    static double ret(double[] c, int n) {
        if (c.length < n + 1) {
            return Double.NaN;
        }
        double past = c[c.length - 1 - n];
        return past > 0 ? c[c.length - 1] / past - 1.0 : Double.NaN;
    }

    /** 12-week momentum skipping the last week — reversal lives in that week. */
    static double ret12wSkip1w(double[] c) {
        if (c.length < 61) {
            return Double.NaN;
        }
        double recent = c[c.length - 1 - 5];
        double past = c[c.length - 1 - 60];
        return past > 0 ? recent / past - 1.0 : Double.NaN;
    }

    /** Percentile rank of {@code value} within {@code all}, in [0,1]. */
    static double pctRank(double value, double[] all) {
        int below = 0, n = 0;
        for (double v : all) {
            if (Double.isNaN(v)) {
                continue;
            }
            n++;
            if (v < value) {
                below++;
            }
        }
        return n == 0 ? 0.5 : (double) below / n;
    }

    /**
     * Rank the universe and return the most oversold names.
     *
     * @param bars   fetched daily series; nulls and short histories are skipped
     * @param topN   how many candidates to return
     */
    static List<Pick> rank(List<MarketData.Bars> bars, Book book,
                           double capital, double riskPct, int topN) {
        List<MarketData.Bars> usable = new ArrayList<>();
        for (MarketData.Bars b : bars) {
            if (b != null && b.size() >= MIN_BARS) {
                usable.add(b);
            }
        }
        if (usable.isEmpty()) {
            return new ArrayList<>();
        }

        // Cross-sectional inputs: a name is only "strong" or "pulled back"
        // relative to what else was available to buy on the same day.
        int n = usable.size();
        double[] mom = new double[n];
        double[] week = new double[n];
        for (int i = 0; i < n; i++) {
            mom[i] = ret12wSkip1w(usable.get(i).close);
            week[i] = ret(usable.get(i).close, 5);
        }

        List<Pick> picks = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            MarketData.Bars b = usable.get(i);
            double a = atr(b.high, b.low, b.close, ATR_PERIOD);
            double r = rsi(b.close, RSI_PERIOD);
            if (Double.isNaN(a) || a <= 0 || Double.isNaN(r)) {
                continue;
            }
            double close = b.lastClose();
            int qty = positionSize(close, a, capital, riskPct, 40.0);
            if (qty <= 0) {
                continue;
            }
            double score;
            if (book.isCommodity()) {
                score = -r;                                   // rsi_oversold
            } else {
                // trend_pullback: in an uptrend by the slow measures, pulled
                // back by the fast one.
                double sma200 = sma(b.close, 200);
                if (Double.isNaN(sma200) || Double.isNaN(mom[i]) || Double.isNaN(week[i])) {
                    continue;
                }
                double aboveTrend = close > sma200 ? 1.0 : 0.0;
                score = aboveTrend * pctRank(mom[i], mom) * (1.0 - pctRank(week[i], week));
            }
            // For trend_pullback a zero score is not a weak buy, it is the rule
            // declining: the name is below its 200-day average. In a broad
            // downtrend every score collapses to zero, the sort becomes
            // arbitrary, and the screen would show a confident "TOP PICK" the
            // strategy never selected. Drop those.
            //
            // The commodity score is -RSI and is legitimately negative, so this
            // test only applies to the equity books.
            if (!book.isCommodity() && !(score > 0)) {
                continue;
            }
            Pick p = new Pick();
            p.symbol = b.symbol;
            p.close = close;
            p.rsi = r;
            p.atr = a;
            p.score = score;
            p.stopDistance = STOP_ATR * a;
            p.quantity = qty;
            p.positionValue = qty * close;
            p.riskRupees = qty * STOP_ATR * a;
            p.costBps = book.isCommodity()
                    ? Costs.commodityRoundTripBps(close, qty, b.symbol)
                    : Costs.deliveryRoundTripBps(close, qty);
            picks.add(p);
        }
        Collections.sort(picks, new Comparator<Pick>() {
            @Override
            public int compare(Pick x, Pick y) {
                return Double.compare(y.score, x.score);      // best first
            }
        });
        return picks.size() > topN ? new ArrayList<>(picks.subList(0, topN)) : picks;
    }
}
