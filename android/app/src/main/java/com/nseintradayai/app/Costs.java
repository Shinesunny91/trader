package com.nseintradayai.app;

/**
 * Groww's charges, ported from {@code src/nse_intraday_ai/costs.py} so the
 * standalone screen quotes the same hurdle the backtest charged.
 *
 * <p>The segment decides almost everything. Equity held overnight settles as
 * DELIVERY, where STT is 0.1% on <em>both</em> legs rather than 0.025% on the
 * sell — a ₹1,00,000 round trip is ~34 bps as CNC against ~13 as MIS, and ₹200
 * of that ₹343 is STT alone. Commodity futures are far cheaper at ~12 bps,
 * which is a large part of why the futures rule clears and the equity one does
 * not.
 *
 * <p>Published rates; verify against a contract note before sizing real money.
 */
final class Costs {

    private static final double BROKERAGE_PCT = 0.001;   // 0.1% of order value
    private static final double BROKERAGE_CAP = 20.0;    // ₹ per executed order
    private static final double BROKERAGE_MIN = 5.0;
    private static final double SEBI_PCT = 0.000001;
    private static final double IPFT_PCT = 0.000001;
    private static final double GST_PCT = 0.18;
    private static final double DP_CHARGE = 20.0;        // delivery sell, per scrip

    // commodity futures (MCX)
    private static final double CTT_SELL = 0.0001;       // 0.01%, non-agri only
    private static final double MCX_TXN = 0.000026;
    private static final double MCX_STAMP_BUY = 0.00002;

    // equity delivery (CNC)
    private static final double STT_DELIVERY = 0.001;    // 0.1%, BOTH legs
    private static final double NSE_TXN = 0.0000297;
    private static final double STAMP_BUY_DELIVERY = 0.00015;

    private static final double SLIPPAGE_BPS_PER_LEG = 5.0;

    private Costs() {
    }

    private static double brokerage(double turnover) {
        if (turnover <= 0) {
            return 0;
        }
        return Math.min(BROKERAGE_CAP, Math.max(BROKERAGE_MIN, turnover * BROKERAGE_PCT));
    }

    /** Round-trip cost in bps of position value, assuming a flat exit price. */
    static double commodityRoundTripBps(double price, int quantity, String symbol) {
        if (price <= 0 || quantity <= 0) {
            return 0;
        }
        double leg = price * quantity;
        double total = leg * 2;
        double ctt = MarketData.isAgri(symbol) ? 0.0 : leg * CTT_SELL;
        double broker = brokerage(leg) * 2;
        double txn = total * MCX_TXN;
        double sebi = total * (SEBI_PCT + IPFT_PCT);
        double stamp = leg * MCX_STAMP_BUY;
        double gst = (broker + txn + sebi) * GST_PCT;
        double slip = total * SLIPPAGE_BPS_PER_LEG / 1e4;
        return (broker + ctt + txn + sebi + stamp + gst + slip) / leg * 1e4;
    }

    /** Equity held overnight — the expensive one. */
    static double deliveryRoundTripBps(double price, int quantity) {
        if (price <= 0 || quantity <= 0) {
            return 0;
        }
        double leg = price * quantity;
        double total = leg * 2;
        double broker = brokerage(leg) * 2;
        double stt = total * STT_DELIVERY;
        double txn = total * NSE_TXN;
        double sebi = total * (SEBI_PCT + IPFT_PCT);
        double stamp = leg * STAMP_BUY_DELIVERY;
        double gst = (broker + txn + sebi + DP_CHARGE) * GST_PCT;
        double slip = total * SLIPPAGE_BPS_PER_LEG / 1e4;
        return (broker + DP_CHARGE + stt + txn + sebi + stamp + gst + slip) / leg * 1e4;
    }
}
