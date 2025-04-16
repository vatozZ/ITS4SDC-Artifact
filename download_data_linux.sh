#!/usr/bin/env bash

echo "The dataset is retrieving from Zenodo ..."

if [ -f "data/data.rar" ]; then
    echo "The dataset (data/data.rar) already exists. Skipping download."
else
    curl -L "https://zenodo.org/api/records/14599223/files/executed-10000.rar/content" -o data/data.rar
    echo "Dataset downloaded successfully!"
fi


if ! [[ -d "data/executed-10000" ]]; then

  echo "Extracting archive..."

  OS=$(uname)

  if [[ "$OS" == "Linux" || "$OS" == "Darwin" ]]; then

      if command -v unrar >/dev/null 2>&1; then
          unrar x -o+ data/data.rar data/
      else
          echo "'unrar' not found. Installing..."
          sudo apt install -y unrar
          exit 1
      fi

  elif [[ "$OS" == "MINGW64_NT"* || "$OS" == "CYGWIN"* ]]; then

      "/c/Program Files/WinRAR/WinRAR.exe" x -o+ data/data.rar data/

  else
      echo "Unsupported OS: $OS"
      echo -e "Please download the dataset manually:\nhttps://zenodo.org/api/records/14599223/files/executed-10000.rar/content"
      exit 1
  fi

fi

echo "The file was extracted successfully!"

echo Downloading test data from GitHub...

mkdir data/test_data
cd data/test_data

git clone https://github.com/christianbirchler-org/sdc-testing-competition.git temp_repo
cd temp_repo
git lfs pull

cp evaluator/sample_tests/sdc-test-data.json ../
cd ..
rm -rf temp_repo

echo Test data downloaded successfully!

