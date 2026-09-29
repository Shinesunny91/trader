#!/usr/bin/env bash
set -euo pipefail

ANDROID_DIR="$(cd "$(dirname "$0")" && pwd)"
SDK_ROOT="$ANDROID_DIR/.android-sdk"
APK_PATH="$ANDROID_DIR/nse-signal-lab-debug.apk"

if [ ! -f "$APK_PATH" ]; then
    "$ANDROID_DIR/build_apk.sh"
fi

ADB_BIN="${ADB:-}"
if [ -z "$ADB_BIN" ]; then
    if [ -x "$SDK_ROOT/platform-tools/adb" ]; then
        ADB_BIN="$SDK_ROOT/platform-tools/adb"
    elif command -v adb >/dev/null 2>&1; then
        ADB_BIN="$(command -v adb)"
    else
        "$ANDROID_DIR/build_apk.sh"
        ADB_BIN="$SDK_ROOT/platform-tools/adb"
    fi
fi

if [ ! -x "$ADB_BIN" ]; then
    echo "adb was not found. Build the APK, then install it manually from:"
    echo "  $APK_PATH"
    exit 1
fi

echo "Waiting for an Android device or emulator..."
"$ADB_BIN" wait-for-device

echo "Installing $APK_PATH"
"$ADB_BIN" install -r "$APK_PATH"

echo "Launching NSE Signal Lab"
"$ADB_BIN" shell am start -n com.nseintradayai.app/.MainActivity >/dev/null
