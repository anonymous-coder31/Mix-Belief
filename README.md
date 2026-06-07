# Mix-Belief: Leveraging Belief Masses for Uncertainty-Guided Mixup in Class-Imbalanced Datasets

This repository implements a belief-driven curriculum learning framework for imbalanced text classification. It combines *Mixup-style data augmentation* with *Evidential Deep Learning (EDL)*, guided by epistemic uncertainty estimates and belief mass comparisons.

## 📌 Overview

- Curriculum learning via belief ratio thresholding
- Supports multiple training seeds and robust metric logging
- Flexible configuration via Hydra and YAML
- Implements multiple augmentation modes: `none`, `mixup`, `remix`, `mix-belief`
- Evaluates standard classification metrics, uncertainty, and calibration error

## 🔧 Project Structure
```
├── main.py # Main entry point with Hydra configuration
├── configs/
│ └── default.yaml # Main config file for dataset, model, training
├── models/
│ └── trainers.py # Trainer and CurriculumTrainer classes
│ └── data_confings.py # Configuration of datasets
│ └── data.py # Loading and Preprocessing the datasets
│ └── losses.py # Combined loss
│ └── models.py # Bert model and textbert classes with mixup
│ └── utils.py # compute_metrics, create_imbalance, get_per to get indexes of samples to interpolate with mixup 
```

## ⚙️ Configuration

All settings are defined in `configs/default.yaml`, including:

- `train.curriculum`: Whether to activate curriculum learning
- `mix.method`: Choice of mixup strategy (`none`, `mixup`, `remix`, `mix-belief`)
- `train.uncertainty`: Whether to enable EDL for uncertainty estimation
- `loss.type`: Loss function used (`CE`, `FL`, `LDAM`)
- `dataset`: Dataset name and imbalance settings
- `train.seeds`: List of seeds for robust multi-run evaluation

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Mix-Belief.git
cd Mix-Belief 
```

### 2.  Install dependencies
```
 pip install -r requirements.txt 
```

### 3.  Run Training

```
    python main.py experiment=my_experiment_name dataset.name=mr 
    mix.method="mix-belief"\
    loss.type="FL"\
    train.uncertainty=true\
    train.curriculum=true
```
### 4. Outputs 
Each experiment creates a folder in results/{dataset}/{experiment}/ with:

`main.log`: Logging file with details of each run

`_results.txt`: Aggregated metrics across seeds, including:

 - F1-score (Macro)
 - Geometric Mean (GM)
 - Mean Epistemic Uncertainty
 - Mean Calibration Error (MCCE)

### Licence
This project is licensed under the **Apache License 2.0**