#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
ANDROID_DIR="$ROOT_DIR/android-app"
APP_DIR="$ANDROID_DIR/app"
SDK_ROOT="$ANDROID_DIR/.android-sdk"
DOWNLOAD_DIR="$ANDROID_DIR/.downloads"
BUILD_DIR="$ANDROID_DIR/manual-build"

CMDLINE_TOOLS_URL="https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip"
ECJ_URL="https://repo.maven.apache.org/maven2/org/eclipse/jdt/ecj/3.42.0/ecj-3.42.0.jar"
JDK_URL="https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse"
JDK_ROOT="$ANDROID_DIR/.jdk"
PACKAGE_NAME="com.nseintradayai.app"
VERSION_CODE="3"
VERSION_NAME="0.3.0"
# Android 11 is API 30.  minSdk 23 means the APK installs on Android 6.0 and
# every release above it, Android 11 included; targetSdk 35 is what Play
# requires and is fully supported on an API 30 device.  Do not raise MIN_SDK
# above 30 to "support Android 11" — that would *drop* everything older.
MIN_SDK="23"
TARGET_SDK="35"

mkdir -p "$SDK_ROOT" "$DOWNLOAD_DIR" "$BUILD_DIR"

ensure_jdk() {
    # The SDK bootstraps itself but still needs a JVM for ecj/apksigner/keytool.
    if command -v java >/dev/null 2>&1 && command -v keytool >/dev/null 2>&1; then
        return
    fi
    local existing
    # bin/java sits at <root>/<jdk-dir>/bin/java — depth 3, not 2.
    existing="$(find "$JDK_ROOT" -maxdepth 3 -name java -type f -perm -u+x 2>/dev/null | head -1)"
    if [ -z "$existing" ]; then
        echo "No system JDK found. Downloading a private Temurin 17..."
        mkdir -p "$JDK_ROOT"
        curl -L --fail --progress-bar "$JDK_URL" -o "$JDK_ROOT/jdk.tar.gz"
        tar xzf "$JDK_ROOT/jdk.tar.gz" -C "$JDK_ROOT"
        rm -f "$JDK_ROOT/jdk.tar.gz"
        existing="$(find "$JDK_ROOT" -maxdepth 3 -name java -type f -perm -u+x | head -1)"
    fi
    if [ -z "$existing" ]; then
        echo "Could not provision a JDK; install one and re-run." >&2
        exit 1
    fi
    JAVA_HOME="$(cd "$(dirname "$(dirname "$existing")")" && pwd)"
    export JAVA_HOME
    export PATH="$JAVA_HOME/bin:$PATH"
    echo "Using JDK at $JAVA_HOME"
}

ensure_cmdline_tools() {
    local sdkmanager="$SDK_ROOT/cmdline-tools/latest/bin/sdkmanager"
    if [ -x "$sdkmanager" ]; then
        return
    fi

    local zip_path="$DOWNLOAD_DIR/commandlinetools-linux.zip"
    local tmp_dir
    tmp_dir="$(mktemp -d)"

    echo "Downloading Android command line tools..."
    curl -L --fail --continue-at - "$CMDLINE_TOOLS_URL" -o "$zip_path"
    unzip -q -o "$zip_path" -d "$tmp_dir"

    rm -rf "$SDK_ROOT/cmdline-tools/latest"
    mkdir -p "$SDK_ROOT/cmdline-tools"
    mv "$tmp_dir/cmdline-tools" "$SDK_ROOT/cmdline-tools/latest"
    rm -rf "$tmp_dir"
}

ensure_android_packages() {
    local sdkmanager="$SDK_ROOT/cmdline-tools/latest/bin/sdkmanager"
    local aapt2="$SDK_ROOT/build-tools/35.0.0/aapt2"
    local android_jar="$SDK_ROOT/platforms/android-35/android.jar"
    local adb="$SDK_ROOT/platform-tools/adb"

    if [ -x "$aapt2" ] && [ -f "$android_jar" ] && [ -x "$adb" ]; then
        return
    fi

    echo "Accepting Android SDK licenses..."
    (yes || true) | "$sdkmanager" --sdk_root="$SDK_ROOT" --licenses >/dev/null

    echo "Installing Android SDK packages..."
    "$sdkmanager" --sdk_root="$SDK_ROOT" \
        "platform-tools" \
        "platforms;android-35" \
        "build-tools;35.0.0"
}

ensure_ecj() {
    local ecj_jar="$DOWNLOAD_DIR/ecj.jar"
    if [ -s "$ecj_jar" ]; then
        return
    fi

    echo "Downloading Eclipse Java compiler..."
    curl -L --fail "$ECJ_URL" -o "$ecj_jar"
}

prepare_dirs() {
    rm -rf "$BUILD_DIR"
    mkdir -p \
        "$BUILD_DIR/classes" \
        "$BUILD_DIR/dex" \
        "$BUILD_DIR/generated" \
        "$BUILD_DIR/resources"
}

prepare_manifest() {
    sed \
        "s|<manifest xmlns:android=\"http://schemas.android.com/apk/res/android\">|<manifest xmlns:android=\"http://schemas.android.com/apk/res/android\" package=\"$PACKAGE_NAME\">|" \
        "$APP_DIR/src/main/AndroidManifest.xml" > "$BUILD_DIR/AndroidManifest.xml"
}

compile_resources() {
    local aapt2="$SDK_ROOT/build-tools/35.0.0/aapt2"
    local android_jar="$SDK_ROOT/platforms/android-35/android.jar"

    "$aapt2" compile \
        --dir "$APP_DIR/src/main/res" \
        -o "$BUILD_DIR/resources/compiled.zip"

    "$aapt2" link \
        -I "$android_jar" \
        --manifest "$BUILD_DIR/AndroidManifest.xml" \
        --java "$BUILD_DIR/generated" \
        --min-sdk-version "$MIN_SDK" \
        --target-sdk-version "$TARGET_SDK" \
        --version-code "$VERSION_CODE" \
        --version-name "$VERSION_NAME" \
        --auto-add-overlay \
        -o "$BUILD_DIR/app-unsigned.apk" \
        "$BUILD_DIR/resources/compiled.zip"
}

compile_java() {
    local android_jar="$SDK_ROOT/platforms/android-35/android.jar"
    local ecj_jar="$DOWNLOAD_DIR/ecj.jar"

    mapfile -t java_sources < <(find "$BUILD_DIR/generated" "$APP_DIR/src/main/java" -name '*.java' | sort)
    java -jar "$ecj_jar" \
        -nowarn \
        -source 1.8 \
        -target 1.8 \
        -bootclasspath "$android_jar" \
        -classpath "$android_jar" \
        -d "$BUILD_DIR/classes" \
        "${java_sources[@]}"
}

dex_classes() {
    local d8="$SDK_ROOT/build-tools/35.0.0/d8"
    local android_jar="$SDK_ROOT/platforms/android-35/android.jar"

    mapfile -t class_files < <(find "$BUILD_DIR/classes" -name '*.class' | sort)
    "$d8" \
        --min-api "$MIN_SDK" \
        --lib "$android_jar" \
        --output "$BUILD_DIR/dex" \
        "${class_files[@]}"

    (cd "$BUILD_DIR/dex" && zip -q "$BUILD_DIR/app-unsigned.apk" classes.dex)
}

sign_apk() {
    local zipalign="$SDK_ROOT/build-tools/35.0.0/zipalign"
    local apksigner="$SDK_ROOT/build-tools/35.0.0/apksigner"
    local keystore="$ANDROID_DIR/debug.keystore"
    local apk_target="$ROOT_DIR/nse-signal-lab-debug.apk"

    if [ ! -f "$keystore" ]; then
        keytool -genkeypair \
            -keystore "$keystore" \
            -storepass android \
            -keypass android \
            -alias androiddebugkey \
            -keyalg RSA \
            -keysize 2048 \
            -validity 10000 \
            -dname "CN=Android Debug,O=Android,C=US" >/dev/null
    fi

    "$zipalign" -f 4 "$BUILD_DIR/app-unsigned.apk" "$BUILD_DIR/app-aligned.apk"
    "$apksigner" sign \
        --ks "$keystore" \
        --ks-pass pass:android \
        --key-pass pass:android \
        --out "$apk_target" \
        "$BUILD_DIR/app-aligned.apk"
    "$apksigner" verify --verbose "$apk_target"

    echo "APK ready: $apk_target"
}

ensure_jdk
ensure_cmdline_tools
ensure_android_packages
ensure_ecj
prepare_dirs
prepare_manifest
compile_resources
compile_java
dex_classes
sign_apk
