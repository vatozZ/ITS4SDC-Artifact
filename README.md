# ITS4SDC-Artifact

This repository contains the **replication package** for the paper

> **"ITS4SDC at the ICST 2025 Tool Competition – Self-Driving Car Testing Track"** <br/> 
> Submitted to ICST 2025 Self-Driving Car Testing Track.

It includes:
- The source code of the tool,
- Scripts for dataset preparation and preprocessing,
- Configuration files for training
- Model training and evaluation logic.

---

## Setup Instructions

### 1. Clone the repository
```
git clone https://github.com/vatozZ/ITS4SDC-Artifact.git
cd ITS4SDC-Artifact
```

### 2. Download and extract the dataset
On a Windows machine,<br/> 
double click to **download_data_windows** file, it will automatically download the dataset. <br/> <br/>
On a Linux machine, <br/>
open the terminal inside of the **ITS4SDC-Artifact** folder and run following command:
```bash
bash download_data_linux.sh
```
Make sure you have ```curl```. If a compatible extractor (e.g. unrar or WinRAR) is not installed, it will install the **unrar** package.

### (Recommended) Create and activate a virtual environment
```
python -m pip install --upgrade pip # upgrade pip
python -m venv its4sdc_venv 
its4sdc_venv\Scripts\activate # (Windows)
source its4sdc_venv/bin/activate # (Linux/Mac)

```

### 3. Install the dependencies 
```
python -m pip install --upgrade pip

pip install -r requirements.txt
```


### 4. Run the Experiment
To modify the number of epochs or other model hyperparameters prior to training or testing, edit the configuration file located at ```experiments/configs/config.yaml```. 
The following command will first merge the dataset into a single JSON file (```dataset_combined.json```) and save it under the ```data/``` directory for later use. 
Then, it will train the LSTM network using that file. If you provide the ```--test_file``` argument for the test file, it will also evaluate performance metrics (accuracy, precision, recall, and F1-score) on the given test file.
For full training (utilizes the full dataset):
```
python src/run.py
```

For cross-validation change the following parameters from the  ```config.yaml``` file and run the program again:
```
training_mode: crossvalidate
k_fold: 10
```

### Outputs
- Trained models: ```data/saved_models/```
- Confusion matrices: ```data/saved_models/confusion_matrices```
- Metrics: ```cross_validation_results.csv```

### Dataset Information
The dataset is available on Zenodo:
https://zenodo.org/records/14599223 <br/>
Size ~ 329.2 MB
Format: ```.json``` files per test case with road points and test outcomes (PASS/FAIL) for ETK-800 model vehicle with the following parameters:
- Risk Factor: 1.5
- Out-of-Bound Tolerance: 50%
- Maximum Speed: 120 km/h

If you use this tool or dataset, please cite the paper:

> @inproceedings{its4sdc2025, <br/>
title={ITS4SDC at the ICST 2025 Tool Competition – Self-Driving Car Testing Track}, <br/>
author={Gullu A.; Shah F. A.; Pfahl D.}, <br/>
booktitle={18th IEEE International Conference on Software Testing, Verification and Validation (ICST)}, <br/>
publisher={IEEE}, <br/>
city={Naples, Italy}, <br/>
month={April}, <br/>
year={2025}<br/>
}

For questions, please contact: *ali.ihsan.gullu@ut.ee*







