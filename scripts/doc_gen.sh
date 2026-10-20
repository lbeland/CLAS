#!/bin/sh

THEME_VERSION="v2.5.0"
THEME_NUM=$(echo "$THEME_VERSION" | sed 's/^v//')
TARGET_DIR="build/doxygen-awesome-css"
PKG_JSON="$TARGET_DIR/package.json"
SHOULD_DOWNLOAD=1
DOXY_VERSION="1.18.0"
INITIAL_DIR=$(pwd)

CURRENT_DOXY_VER=$(doxygen --version 2>/dev/null | awk '{print $1}')

mkdir -p build

sudo apt-get update
sudo apt-get install -y doxygen graphviz

if [ "$CURRENT_DOXY_VER" != "1.18.0" ]; then
    echo "Doxygen 1.18.0 not found (Found: ${CURRENT_DOXY_VER:-none}). Installing..."

    TMP_SETUP="/tmp/doxygen-setup-$$"
    rm -rf "$TMP_SETUP"
    mkdir -p "$TMP_SETUP"

    wget -P "$TMP_SETUP"/ https://github.com/doxygen/doxygen/releases/download/Release_1_18_0/doxygen-1.18.0.linux.bin.tar.gz
    if [ $? -ne 0 ]; then
        echo "Error: Doxygen download failed."
        exit 1
    fi

    cd "$TMP_SETUP"
    tar -xf doxygen-1.18.0.linux.bin.tar.gz
    cd doxygen-1.18.0
    sudo make install

    cd "$INITIAL_DIR"
    rm -rf "$TMP_SETUP"
else
    echo "Doxygen 1.18.0 is already installed."
fi

if [ -f "$PKG_JSON" ]; then
    CURRENT_VERSION=$(grep '"version"' "$PKG_JSON" | cut -d'"' -f4)
    if [ "$CURRENT_VERSION" = "$THEME_NUM" ]; then
        echo "Theme version ${THEME_VERSION} already installed. Skipping download."
        SHOULD_DOWNLOAD=0
    fi
fi

if [ "$SHOULD_DOWNLOAD" -eq 1 ]; then
    echo "Downloading Doxygen Awesome Theme ${THEME_VERSION}..."
    rm -rf "$TARGET_DIR" build/html

    wget -P build/ https://github.com/jothepro/doxygen-awesome-css/archive/refs/tags/${THEME_VERSION}.zip
    if [ $? -ne 0 ]; then
        echo "Error: Download failed."
        exit 1
    fi

    unzip -q build/${THEME_VERSION}.zip -d build/
    mv build/doxygen-awesome-css-${THEME_NUM} "$TARGET_DIR"

    rm build/${THEME_VERSION}.zip
fi

mkdir -p build

echo "Running Doxygen..."

doxygen -q docs/doxygen/Doxyfile
