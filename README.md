# TKGL

**Toxicological Knowledge–guided Graph-based Learning (TKGL)** is a multi-view learning framework that advances graph-based representation learning by integrating substructural alerts (SAs) and biological mechanism information for molecular toxicity prediction.


## Installation

### 1. Clone the Repository

First, clone the project repository from GitHub to your local machine by running the following command in your terminal:

```bash
git clone https://github.com/yanjing-duan/TKGL.git
```

This command will create a copy of the project in your current working directory.


### 2. Set Up the Environment

After cloning the repository, navigate to the project directory and create the required Conda environments.

This project uses **two separate environments**:

* One for fingerprint generation during data preprocessing
* One for toxicity prediction model training

Run the following commands:

```bash
conda env create -f environment_tkgl-fps.yml
conda env create -f environment_tkgl.yml
```

These commands will create Conda environments according to the specifications in the respective `.yml` files and install all required dependencies.


## Data Preparation

### 0. Activate the Environment (Fingerprint Generation)

```bash
conda activate tkgl-fps
```


### 1. SA Fingerprint Generation (for Specific Toxicity Datasets)

Before running, specify the dataset names in the `dataset_list` variable within the script.

Step 1: Identify structural alerts (SA):

```bash
python ./data_process/extract_cal_tox_sub.py
```

Step 2: Generate SA fingerprints:

```bash
python ./data_process/generate_cal_sub_fp.py
```


### 2. Bioassay Fingerprint Generation

This step computes Bioassay fingerprints:

Before running, specify the dataset names in the `dataset_list` variable within the script.

```bash
python ./data_process/generate_cal_bioassay_fp.py
```


### 3. Merge Bioassay and SA Fingerprints

Before running, specify the dataset names in the `dataset_list` variable within the script.

```bash
python ./data_process/merge_bioassay_and_SA_fp.py
```


## Toxicity Prediction Model Training and Evaluation

### 0. Activate the Training Environment

```bash
conda activate tkgl
```


### 1. Self-supervised Pretraining (Graph-based Encoder for TKGL)

```bash
python ./TKGL_pretrain.py
```


### 2. TKGL Training and Evaluation (Customizable Hyperparameters)

```bash
python ./TKGL_main.py --task "task_name"
```

You can modify additional hyperparameters inside the script or via command-line arguments as needed.

