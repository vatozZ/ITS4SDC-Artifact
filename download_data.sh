#!/usr/bin/env bash

echo "The dataset is retrieving from Zenodo ..."

curl -L "https://zenodo.org/api/records/14599223/files/executed-10000.rar/content" -o data/data.rar

echo "Extracting archive..."

OS=$(uname)

if [[ "$OS" == "Linux" || "$OS" == "Darwin" ]]; then

    if command -v unrar >/dev/null 2>&1; then
        unrar x -o+ data/data.rar data/
    else
        echo "'unrar' not found. Please install it:"
        echo "sudo apt install unrar"
        sudo apt install unrar
        exit 1
    fi

elif [[ "$OS" == "MINGW64_NT"* || "$OS" == "CYGWIN"* ]]; then

    "/c/Program Files/WinRAR/WinRAR.exe" x -o+ data/data.rar data/

else
    echo "Unsupported OS: $OS"
    echo -e "Please download the dataset manually:\nhttps://zenodo.org/api/records/14599223/files/executed-10000.rar/content"
    exit 1
fi

echo "The file was extracted successfully!"
