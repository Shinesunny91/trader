package com.nseintradayai.app;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Typeface;
import android.net.ConnectivityManager;
import android.net.Network;
import android.net.NetworkCapabilities;
import android.net.NetworkInfo;
import android.os.Build;
import android.os.Bundle;
import android.view.Gravity;
import android.view.KeyEvent;
import android.view.View;
import android.view.inputmethod.EditorInfo;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;

public class MainActivity extends Activity {
    private static final String PREFS_NAME = Storage.PREFS;
    static final String SERVER_URL_KEY = "server_url";
    static final String DEFAULT_SERVER_URL = "http://10.0.2.2:8501";
    /** Port `scripts/ticket_api.py` listens on, beside Streamlit's 8501. */
    static final int API_PORT = 8502;
    private static final int REQ_POST_NOTIFICATIONS = 91;

    private TextView storageText;
    private CheckBox notifyToggle;
    private CheckBox sdToggle;

    private EditText serverInput;
    private ProgressBar progressBar;
    private TextView statusText;
    private WebView webView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        buildLayout();
        configureWebView();

        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        String serverUrl = prefs.getString(SERVER_URL_KEY, DEFAULT_SERVER_URL);
        serverInput.setText(serverUrl);
        loadServer(serverUrl);

        TicketJobService.ensureChannel(this);
        boolean notify = prefs.getBoolean(TicketJobService.KEY_NOTIFY, true);
        notifyToggle.setChecked(notify);
        sdToggle.setChecked(Storage.preferSdCard(this));
        refreshStorageText();
        if (notify) {
            requestNotificationPermissionIfNeeded();
            TicketJobService.schedule(this, true);
        }
    }

    /**
     * POST_NOTIFICATIONS only exists from API 33. On Android 11 the permission
     * does not exist and notifications are granted by default, so this is a
     * no-op there — but the app targets 35, so a 13+ device must be asked.
     */
    private void requestNotificationPermissionIfNeeded() {
        if (Build.VERSION.SDK_INT < 33) {
            return;
        }
        if (checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(
                    new String[]{android.Manifest.permission.POST_NOTIFICATIONS},
                    REQ_POST_NOTIFICATIONS);
        }
    }

    private void refreshStorageText() {
        storageText.setText("Saving to: " + Storage.describe(this));
    }

    @SuppressLint("SetJavaScriptEnabled")
    private void configureWebView() {
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setDatabaseEnabled(true);
        settings.setLoadWithOverviewMode(true);
        settings.setUseWideViewPort(true);
        settings.setSupportZoom(true);
        settings.setBuiltInZoomControls(true);
        settings.setDisplayZoomControls(false);
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_ALWAYS_ALLOW);

        webView.setWebViewClient(new WebViewClient() {
            @Override
            public void onPageStarted(WebView view, String url, android.graphics.Bitmap favicon) {
                progressBar.setVisibility(View.VISIBLE);
                setStatus("Loading " + url, false);
            }

            @Override
            public void onPageFinished(WebView view, String url) {
                progressBar.setVisibility(View.GONE);
                setStatus("Connected to " + url, false);
            }

            @Override
            public void onReceivedError(
                    WebView view,
                    WebResourceRequest request,
                    WebResourceError error
            ) {
                if (request != null && request.isForMainFrame()) {
                    progressBar.setVisibility(View.GONE);
                    setStatus(buildConnectionHelp(), true);
                }
            }
        });

        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public void onProgressChanged(WebView view, int newProgress) {
                progressBar.setProgress(newProgress);
            }
        });
    }

    private void buildLayout() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.WHITE);
        root.setLayoutParams(new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.MATCH_PARENT
        ));

        TextView title = new TextView(this);
        title.setText(getString(R.string.app_name));
        title.setTextColor(Color.rgb(15, 23, 42));
        title.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        title.setTextSize(18);
        title.setGravity(Gravity.CENTER_VERTICAL);
        title.setPadding(dp(16), dp(12), dp(16), dp(8));
        root.addView(title, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        ));

        LinearLayout controls = new LinearLayout(this);
        controls.setOrientation(LinearLayout.HORIZONTAL);
        controls.setGravity(Gravity.CENTER_VERTICAL);
        controls.setPadding(dp(12), 0, dp(12), dp(8));

        serverInput = new EditText(this);
        serverInput.setSingleLine(true);
        serverInput.setTextSize(14);
        serverInput.setSelectAllOnFocus(false);
        serverInput.setHint("http://10.0.2.2:8501");
        serverInput.setImeOptions(EditorInfo.IME_ACTION_GO);
        serverInput.setInputType(android.text.InputType.TYPE_TEXT_VARIATION_URI);
        serverInput.setOnEditorActionListener(new TextView.OnEditorActionListener() {
            @Override
            public boolean onEditorAction(TextView view, int actionId, KeyEvent event) {
                boolean enterPressed = event != null
                        && event.getKeyCode() == KeyEvent.KEYCODE_ENTER
                        && event.getAction() == KeyEvent.ACTION_UP;
                if (actionId == EditorInfo.IME_ACTION_GO || enterPressed) {
                    loadServer(serverInput.getText().toString());
                    return true;
                }
                return false;
            }
        });
        controls.addView(serverInput, new LinearLayout.LayoutParams(
                0,
                LinearLayout.LayoutParams.WRAP_CONTENT,
                1f
        ));

        Button connectButton = makeButton("Connect");
        connectButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                loadServer(serverInput.getText().toString());
            }
        });
        controls.addView(connectButton);

        Button reloadButton = makeButton("Reload");
        reloadButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                webView.reload();
            }
        });
        controls.addView(reloadButton);
        root.addView(controls);

        // Standalone mode: works with no laptop on the network at all.
        Button picksButton = makeButton("Standalone picks (works offline)");
        picksButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                startActivity(new android.content.Intent(MainActivity.this, PicksActivity.class));
            }
        });
        LinearLayout.LayoutParams picksParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        picksParams.leftMargin = dp(12);
        picksParams.rightMargin = dp(12);
        root.addView(picksButton, picksParams);

        progressBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progressBar.setMax(100);
        progressBar.setVisibility(View.GONE);
        root.addView(progressBar, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                dp(3)
        ));

        statusText = new TextView(this);
        statusText.setTextColor(Color.rgb(71, 85, 105));
        statusText.setTextSize(12);
        statusText.setPadding(dp(16), dp(6), dp(16), dp(6));
        root.addView(statusText, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        ));

        // ── Settings: where data goes, and whether the phone buzzes ──────
        LinearLayout settings = new LinearLayout(this);
        settings.setOrientation(LinearLayout.VERTICAL);
        settings.setPadding(dp(16), 0, dp(16), dp(6));

        LinearLayout toggles = new LinearLayout(this);
        toggles.setOrientation(LinearLayout.HORIZONTAL);

        sdToggle = new CheckBox(this);
        sdToggle.setText("Save to SD card");
        sdToggle.setTextSize(13);
        sdToggle.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton view, boolean checked) {
                Storage.setPreferSdCard(MainActivity.this, checked);
                refreshStorageText();
            }
        });
        toggles.addView(sdToggle, new LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));

        notifyToggle = new CheckBox(this);
        notifyToggle.setText("Notify on new ticket");
        notifyToggle.setTextSize(13);
        notifyToggle.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton view, boolean checked) {
                getSharedPreferences(PREFS_NAME, MODE_PRIVATE).edit()
                        .putBoolean(TicketJobService.KEY_NOTIFY, checked).apply();
                if (checked) {
                    requestNotificationPermissionIfNeeded();
                }
                TicketJobService.schedule(MainActivity.this, checked);
                setStatus(checked
                        ? "Checking for new tickets every 15 minutes."
                        : "Notifications off.", false);
            }
        });
        toggles.addView(notifyToggle, new LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        settings.addView(toggles);

        storageText = new TextView(this);
        storageText.setTextColor(Color.rgb(100, 116, 139));
        storageText.setTextSize(11);
        settings.addView(storageText);
        root.addView(settings);

        FrameLayout webContainer = new FrameLayout(this);
        webView = new WebView(this);
        webContainer.addView(webView, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
        ));
        root.addView(webContainer, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                0,
                1f
        ));

        setContentView(root);
    }

    private Button makeButton(String label) {
        Button button = new Button(this);
        button.setText(label);
        button.setAllCaps(false);
        button.setTextSize(13);
        button.setMinHeight(dp(44));
        button.setMinWidth(dp(84));
        button.setPadding(dp(8), 0, dp(8), 0);
        return button;
    }

    private void loadServer(String rawUrl) {
        String normalizedUrl = normalizeUrl(rawUrl);
        serverInput.setText(normalizedUrl);
        getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
                .edit()
                .putString(SERVER_URL_KEY, normalizedUrl)
                .apply();

        if (!isNetworkAvailable()) {
            setStatus("No active network. Connect to the same Wi-Fi as the Streamlit server.", true);
        } else {
            setStatus("Opening " + normalizedUrl, false);
        }

        webView.loadUrl(normalizedUrl);
    }

    private String normalizeUrl(String rawUrl) {
        String value = rawUrl == null ? "" : rawUrl.trim();
        if (value.isEmpty()) {
            value = DEFAULT_SERVER_URL;
        }
        if (!value.startsWith("http://") && !value.startsWith("https://")) {
            value = "http://" + value;
        }
        return value;
    }

    private void setStatus(String text, boolean error) {
        statusText.setText(text);
        statusText.setTextColor(error ? Color.rgb(185, 28, 28) : Color.rgb(71, 85, 105));
    }

    private String buildConnectionHelp() {
        return "Cannot reach the trading app. Start ./run_android_server.sh on the computer, "
                + "then enter the printed phone URL here. Emulator default: "
                + DEFAULT_SERVER_URL + ".";
    }

    /**
     * Android 11 (API 30) reports a hardcoded, always-"connected" NetworkInfo to
     * apps targeting API 29+, so the legacy call could not detect a phone that
     * had dropped off Wi-Fi — the user got a blank WebView instead of the
     * "connect to the same Wi-Fi" hint. NetworkCapabilities is the API that
     * still answers truthfully; the old path stays for API 22 and below.
     */
    @SuppressWarnings("deprecation")
    private boolean isNetworkAvailable() {
        ConnectivityManager manager = (ConnectivityManager) getSystemService(Context.CONNECTIVITY_SERVICE);
        if (manager == null) {
            return true;
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            Network active = manager.getActiveNetwork();
            if (active == null) {
                return false;
            }
            NetworkCapabilities caps = manager.getNetworkCapabilities(active);
            return caps != null
                    && caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET);
        }
        NetworkInfo networkInfo = manager.getActiveNetworkInfo();
        return networkInfo != null && networkInfo.isConnected();
    }

    private int dp(int value) {
        return (int) (value * getResources().getDisplayMetrics().density + 0.5f);
    }

    @Override
    public void onBackPressed() {
        if (webView != null && webView.canGoBack()) {
            webView.goBack();
            return;
        }
        super.onBackPressed();
    }
}
