# Geometric Attention Is All You Need
*Project repository to accompany the poster and article [Graph Structure From Point Clouds: Geometric Attention Is All You Need](https://neurips.cc/virtual/2022/workshop/49979#wse-detail-56982)*

**Note: This repository is a work in progress until December 3rd, 2022, when it will be fully supported for reproducibility of paper/poster results**

## Repository Structure
All scripts to run from the commandline are available at the top-level: `preprocess_events.py`, `train_model.py` and `evaluate_model.py`. These scripts call upon config files in the `Config` directory. They use the Pytorch Lightning modules in the `LightningModules` directory. In that directory, `jet_gnn_base.py` contains a Lightning class that handles all training logic. The `Models/gravnet.py` contains our implementation of the `GravNet` class, which can handle both the vanilla GravNet convolution, and our model GravNetNorm.

## Reproduce Paper Results

### 1. Setup

First, install dependencies with conda:
```bash
conda env create -f gpu_environment.yml
conda activate geometric-attention
pip install -r requirements.txt
```
The following instructions assume a GPU is available, and a host device with around 50Gb of RAM. For logging, we use Weights and Biases. If you wish to use this, you should create an account and `pip install wandb`. 

### 2. Download Data

Download the `train.h5, val.h5, test.h5` files from the (Top Quark Tagging Reference Dataset)[https://zenodo.org/record/2603256#.Y4idbXZKgQ8]. Place these files in your data directory `MY_DATA_PATH/raw_input`.

### 3. Preprocess Data

Preprocess the data with:
```bash
python preprocess_events.py Configs/preprocess_config.yaml
```
This can take around 20 minutes (it is done on a single thread for now).

### 4. Train Model

Train the model with:
```bash
python train_model.py Configs/small_train_norm.yaml
```

This will train the GravNetNorm model on a small subset of the full dataset. This is useful to ensure that the model is training correctly. To train with the full dataset, use the `full_train_norm.yaml` config file.

Additionally, one can compare with the vanilla GravNet model by using the `full_train_vanilla.yaml` config file.

### 5. Evaluate Model

Evaluate the model with:
```bash
python evaluate_model.py Configs/small_train_norm.yaml
```

This will run the best model checkpoint (as determined by validation AUC) on the test dataset, and output the test AUC, accuracy and background rejection rate to the console.

## Citation
If you use this code in your research, for now please cite the conference submission:
```
@inproceedings{geometric-attention,
  title={Graph Structure From Point Clouds: Geometric Attention Is All You Need},
  author={Daniel Murnane},
  booktitle={NeurIPS Workshop on Graph Representation Learning},
  year={2022}
}
```
