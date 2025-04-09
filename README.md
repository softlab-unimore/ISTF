# Code for paper "Forecasting Irregularly Sampled Time Series with Transformer Encoders"

📄 **[Paper Abstract](#abstract)**  
📊 **[Datasets](#datasets)**  
🚀 **[Usage](#usage)**  
⚙️ **[Command-line Arguments](#command-line-arguments)**

---

## 📄 Paper Abstract

Time series forecasting is a fundamental task in various domains, including environmental monitoring, finance, and healthcare. State-of-the-art forecasting models typically assume that time series are uniformly sampled. However, in real-world scenarios, data is often collected at irregular intervals and with missing values, due to sensor failures or network issues. This makes traditional forecasting approaches unsuitable.

In this paper, we introduce **ISTF** (Irregular Sequence Transformer Forecasting), a novel transformer-based architecture designed for forecasting irregularly sampled multivariate time series. ISTF leverages exogenous variables as contextual information to enhance the prediction of a single target variable. The architecture first regularizes the MTS on a fixed temporal scale, keeping track of missing values. Then, a dedicated embedding strategy, based on a local and global attention mechanism, aims at capturing dependencies between timestamps, sources and missing values. We evaluate ISTF on two real-world datasets, **FrenchPiezo** and **USHCN**. The experimental results demonstrate that ISTF outperforms competing approaches in forecasting accuracy while remaining computationally efficient.

---

## 📊 Datasets

The repository contains the smaller versions of the datasets, useful for debugging. The full datasets can be downloaded following the indications below.

### FrenchPiezo
Download `dataset_2015_2021.csv` and `dataset_stations.csv` from https://zenodo.org/records/7193812 and place them in `data/FrenchPiezo/`.

### USHCN
Run `ushcn_preprocessing.py` to download and preprocess the USHCN dataset. The file `pivot_1990_1993_spatial.csv` will be placed in `data/USHCN/`. The script is adapted from https://github.com/bw-park/ACSSM/blob/main/lib/ushcn_preprocessing.py

---

## 🚀 Usage

To preprocess the data and store it in a `.pickle`, run:

```bash
python data_step.py --dataset french --nan-percentage 0.2 --num-past 48 --num-future 60
```

To preprocess the data, train the model and test it, run:

```bash
python pipeline.py --dataset french --nan-percentage 0.2 --num-past 48 --num-future 60
```

To evaluate ablations or adjust hyperparameters, modify the command-line arguments accordingly.

---

## ⚙️ Command-line Arguments

### Dataset and Preprocessing
- `--dataset`: Dataset to use [`french`, `ushcn`] (**required**)
- `--nan-percentage`: Percentage of missing values to simulate (**required**)
- `--num-past`: Number of past timesteps for input (**required**)
- `--num-future`: Number of future timesteps to forecast (**required**)
- `--scaler-type`: Scaling method [`standard`, `minmax`] (default: `standard`)

### Model Architecture
- `--kernel-size`: Kernel size for convolutional embedder (default: `5`)
- `--d-model`: Embedding dimension (default: `32`)
- `--num-heads`: Attention heads in Transformer (default: `4`)
- `--dff`: Hidden dimension of Transformer FFN (default: `64`)
- `--num-layers`: Number of Transformer encoder layers (default: `2`)
- `--gru`: Number of units in the GRU layer (default: `64`)
- `--fff`: Feed-forward network sizes post-Transformer (default: `[128]`)
- `--l2-reg`: L2 regularization (default: `1e-2`)
- `--dropout`: Dropout rate (default: `0.1`)
- `--activation`: Activation function (default: `relu`)

### Optimization
- `--lr`: Learning rate (default: `1e-4`)
- `--loss`: Loss function (default: `mse`)
- `--batch-size`: Batch size (default: `64`)
- `--epochs`: Number of training epochs (default: `100`)
- `--patience`: Early stopping patience (default: `20`, `-1` disables)

### Ablations
- `--no-embedder`: Disable convolutional embedder
- `--no-local-global`: Disable local-global attention
- `--no-gru`: Disable GRU

### Misc
- `--force-data-step`: Force regeneration of data (ignore cached pickle)
- `--dev`: Use smaller development data
- `--cpu`: Force CPU usage
- `--seed`: Random seed (default: `42`)
