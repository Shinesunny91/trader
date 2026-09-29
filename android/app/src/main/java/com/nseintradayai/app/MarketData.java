package com.nseintradayai.app;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Daily OHLC straight from Yahoo's chart endpoint, so the app works with no
 * laptop on the network.
 *
 * <p>This is the same free source the Python workspace uses, and it carries the
 * same caveat: it is convenient, not broker-grade. Bars can be late, thin
 * contracts can be missing, and the endpoint occasionally rate-limits. The
 * engine here only makes a decision on <em>closed daily bars</em>, so a delayed
 * quote changes nothing — which is exactly why the intra-week rule is the one
 * worth running on a phone, and the 5-minute intraday scanner is not.
 *
 * <p>No API key, no dependency: one GET and org.json, which is in the framework.
 */
final class MarketData {

    /** The commodity universe the ten-year test was run on. */
    static final String[] COMMODITIES = {
            "GC=F", "SI=F", "CL=F", "BZ=F", "RB=F", "HO=F", "NG=F", "HG=F",
            "PL=F", "PA=F", "ZC=F", "ZS=F", "ZW=F", "CT=F", "SB=F",
    };


    /**
     * The 120 most liquid NSE names (median ~Rs 394 cr/day traded), not the
     * full 500.  A full refresh is ~5 MB of JSON; 500 would be ~22 MB, which is
     * not a thing to ask of a phone on mobile data.  Daily bars only change
     * once a day, so a cached refresh costs nothing until the next close.
     */
    static final String[] NSE_LIQUID = {
            "HDFCBANK.NS", "RELIANCE.NS", "ICICIBANK.NS", "INFY.NS",
            "SBIN.NS", "BHARTIARTL.NS", "BSE.NS", "TCS.NS",
            "LT.NS", "ETERNAL.NS", "M&M.NS", "VEDL.NS",
            "MCX.NS", "BAJFINANCE.NS", "AXISBANK.NS", "NETWEB.NS",
            "GROWW.NS", "DIXON.NS", "ADANIPOWER.NS", "KALYANKJIL.NS",
            "KOTAKBANK.NS", "SHRIRAMFIN.NS", "MARUTI.NS", "INDIGO.NS",
            "BEL.NS", "HINDALCO.NS", "TATASTEEL.NS", "HAL.NS",
            "IDEA.NS", "KAYNES.NS", "ITC.NS", "SUNPHARMA.NS",
            "HFCL.NS", "BHEL.NS", "ADANIENT.NS", "HINDCOPPER.NS",
            "COALINDIA.NS", "HCLTECH.NS", "HSCL.NS", "COFORGE.NS",
            "ONGC.NS", "POWERINDIA.NS", "NATIONALUM.NS", "TITAN.NS",
            "TEJASNET.NS", "WAAREEENER.NS", "DATAPATTNS.NS", "TMCV.NS",
            "ADANIPORTS.NS", "TRENT.NS", "ADANIGREEN.NS", "OLAELEC.NS",
            "EICHERMOT.NS", "NTPC.NS", "WIPRO.NS", "ATHERENERG.NS",
            "SUZLON.NS", "HINDUNILVR.NS", "JIOFIN.NS", "ASHOKLEY.NS",
            "PERSISTENT.NS", "TMPV.NS", "SAIL.NS", "TVSMOTOR.NS",
            "TECHM.NS", "SWIGGY.NS", "POWERGRID.NS", "BAJAJ-AUTO.NS",
            "GVT&D.NS", "PAYTM.NS", "ADANIENSOL.NS", "ULTRACEMCO.NS",
            "MAZDOCK.NS", "HINDZINC.NS", "CANBK.NS", "POLYCAB.NS",
            "GRSE.NS", "MEESHO.NS", "POLICYBZR.NS", "HEROMOTOCO.NS",
            "MUTHOOTFIN.NS", "PFC.NS", "APOLLOHOSP.NS", "ATGL.NS",
            "CGPOWER.NS", "BPCL.NS", "CUMMINSIND.NS", "BANKBARODA.NS",
            "VBL.NS", "ASIANPAINT.NS", "MOTHERSON.NS", "IFCI.NS",
            "CHOLAFIN.NS", "HINDPETRO.NS", "UNIONBANK.NS", "ABB.NS",
            "FEDERALBNK.NS", "AMBER.NS", "OFSS.NS", "LENSKART.NS",
            "DRREDDY.NS", "BHARATFORG.NS", "SOLARINDS.NS", "MAXHEALTH.NS",
            "HDFCAMC.NS", "ZEEL.NS", "LAURUSLABS.NS", "DIVISLAB.NS",
            "NESTLEIND.NS", "INDUSINDBK.NS", "LODHA.NS", "INDUSTOWER.NS",
            "DLF.NS", "ANGELONE.NS", "NAUKRI.NS", "IDFCFIRSTB.NS",
            "WOCKPHARMA.NS", "GRASIM.NS", "OIL.NS", "MRPL.NS"
    };

    /** Agri roots are exempt from CTT — it changes the cost, so it is tracked. */
    private static final String AGRI = "ZC ZS ZW CT SB";

    static boolean isAgri(String symbol) {
        return AGRI.contains(symbol.split("=")[0]);
    }

    static final class Bars {
        final String symbol;
        final long[] time;
        final double[] open;
        final double[] high;
        final double[] low;
        final double[] close;

        Bars(String symbol, long[] time, double[] open, double[] high, double[] low, double[] close) {
            this.symbol = symbol;
            this.time = time;
            this.open = open;
            this.high = high;
            this.low = low;
            this.close = close;
        }

        int size() {
            return close.length;
        }

        double lastClose() {
            return close[close.length - 1];
        }
    }

    private MarketData() {
    }

    /**
     * Two years of daily bars — enough for the 200-day trend features with
     * warmup, and ~45 KB per symbol on the wire.
     */
    static Bars fetchDaily(String symbol) throws Exception {
        String url = "https://query1.finance.yahoo.com/v8/finance/chart/"
                + java.net.URLEncoder.encode(symbol, "UTF-8")
                + "?range=2y&interval=1d";
        HttpURLConnection conn = (HttpURLConnection) new URL(url).openConnection();
        conn.setConnectTimeout(10000);
        conn.setReadTimeout(15000);
        // Yahoo refuses the default Java agent.
        conn.setRequestProperty("User-Agent", "Mozilla/5.0 (Android) NSESignalLab");
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
            return parse(symbol, sb.toString());
        } finally {
            conn.disconnect();
        }
    }

    static Bars parse(String symbol, String body) throws Exception {
        JSONObject chart = new JSONObject(body).getJSONObject("chart");
        if (chart.isNull("result")) {
            return null;
        }
        JSONObject result = chart.getJSONArray("result").getJSONObject(0);
        JSONArray stamps = result.getJSONArray("timestamp");
        JSONObject quote = result.getJSONObject("indicators")
                .getJSONArray("quote").getJSONObject(0);
        JSONArray o = quote.getJSONArray("open");
        JSONArray h = quote.getJSONArray("high");
        JSONArray l = quote.getJSONArray("low");
        JSONArray c = quote.getJSONArray("close");

        // Yahoo pads gaps with nulls; a null bar is not a bar.
        List<Integer> keep = new ArrayList<>();
        for (int i = 0; i < stamps.length(); i++) {
            if (!o.isNull(i) && !h.isNull(i) && !l.isNull(i) && !c.isNull(i)) {
                keep.add(i);
            }
        }
        int n = keep.size();
        if (n == 0) {
            return null;
        }
        long[] t = new long[n];
        double[] oo = new double[n], hh = new double[n], ll = new double[n], cc = new double[n];
        for (int k = 0; k < n; k++) {
            int i = keep.get(k);
            t[k] = stamps.getLong(i);
            oo[k] = o.getDouble(i);
            hh[k] = h.getDouble(i);
            ll[k] = l.getDouble(i);
            cc[k] = c.getDouble(i);
        }
        return new Bars(symbol, t, oo, hh, ll, cc);
    }
}
