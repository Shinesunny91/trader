#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
ANDROID_DIR="$ROOT_DIR/android-app"
SDK_ROOT="$ANDROID_DIR/.android-sdk"
APK_PATH="$ROOT_DIR/nse-signal-lab-debug.apk"

if [ ! -f "$APK_PATH" ]; then
    "$ROOT_DIR/build_android_apk.sh"
fi

ADB_BIN="${ADB:-}"
if [ -z "$ADB_BIN" ]; then
    if [ -x "$SDK_ROOT/platform-tools/adb" ]; then
        ADB_BIN="$SDK_ROOT/platform-tools/adb"
    elif command -v adb >/dev/null 2>&1; then
        ADB_BIN="$(command -v adb)"
    else
        "$ROOT_DIR/build_android_apk.sh"
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
