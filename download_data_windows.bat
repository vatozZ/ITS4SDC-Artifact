@echo off
echo Downloading dataset from Zenodo...

curl -L "https://zenodo.org/api/records/14599223/files/executed-10000.rar/content" -o data\data.rar

echo Extracting archive...

if exist "C:\Program Files\WinRAR\WinRAR.exe" (
    "C:\Program Files\WinRAR\WinRAR.exe" x -o+ data\data.rar data\
 ) else (
    echo WinRAR not found at: C:\Program Files\WinRAR\WinRAR.exe
    echo Please install WinRAR and try again.
    exit /b 1
)

echo Extraction the dataset complete!

echo Downloading test data from GitHub...

mkdir data\test_data
cd data\test_data

git clone https://github.com/christianbirchler-org/sdc-testing-competition.git temp_repo
cd temp_repo
git lfs pull

copy evaluator\sample_tests\sdc-test-data.json ..\
cd ..
cd /data/test_data
rmdir /s /q temp_repo

echo Test data downloaded successfully!
