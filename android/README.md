# Android APK

This Android project is a native WebView companion for the Streamlit trading app in this workspace.

The existing Python/Streamlit scanner remains the backend. The APK gives you an installable Android launcher that loads the running Streamlit server.

## Build

From the repository root:

```bash
./android/build_apk.sh
```

The script downloads Android command line tools, Android SDK packages, and the small Eclipse Java compiler into `android/` if they are not installed already (all git-ignored). The built APK is written to:

```text
android/nse-signal-lab-debug.apk
```

## Install

Connect an Android phone with USB debugging enabled, or start an emulator, then run:

```bash
./android/install_apk.sh
```

You can also install the generated APK manually.

## Run The Backend For Android

The ticket API it polls (`/tickets`) serves the gap-reversal book's picks.

The regular `./run_app.sh` binds Streamlit to `127.0.0.1`, which is only visible on this computer. For Android, run:

```bash
./android/run_server.sh
```

Use the printed URL inside the Android app:

- Emulator: `http://10.0.2.2:8501`
- Physical phone: `http://<computer-lan-ip>:8501`

The phone and computer must be on the same network unless you expose the Streamlit server another way.
