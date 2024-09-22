# Towards Holistic Evaluation of Vision Language Histopathology Models
Description to be added later

## Installation
1. Create an environment using the provided .yml file
```
conda env create -f environment.yml
conda activate benchmark
cd CONCH
pip install --upgrade pip
pip install -e .
```
2. Add the relative paths in :\\
    a. The paths to the data, caching and results folder in the dotenv file : "plip/reproducibility/config_example.env"
    b. The paths to the CONCH and MI-Zero models in : "plip/reproducibility/factory.py"
    c. Path to MI-Zero configuration path in : "plip/src/models/factory.py"

## Running Zero-shot
Run the following bash file in "plip/reproducibility" :
```
bash zero_shot.sh
```