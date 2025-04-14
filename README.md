# ITS4SDC-Artifact

This repository contains the **replication package** for the paper

> **"ITS4SDC at the ICST 2025 Tool Competition – Self-Driving Car Testing Track"** <br/> 
> Submitted to ICST 2025 Self-Driving Car Testing Track.

It includes:
- The source code of the tool,
- Scripts for dataset preperation and preprocessing,
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
```
bash download_data.sh
```
Make sure you have ```curl``` and a compatäble extractor (e-g- unrar or WinRAR) installed.

### 3. Create a virtual environment (recommended)
```
python -m venv its4sdc python==???
```

### 4. Run the Experiment
For full training (utilizes the full dataset):
```
python src/run.py
```

For cross-validation, in ```config.yaml```, change:
```
training_mode: crossvalidate
k_fold: 10
```

Then run: 

```
python src/run.py
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
>- Risk Factor: 1.5
>- Out-of-Bound Tolerance: %50
>- Maximum Speed: 120 km/h

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

For questions, please contact:
>ali.ihsan.gullu@ut.ee







