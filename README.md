# Refining GPT-3 Embeddings with a Siamese Structure for Technical Post Duplicate Detection

## Datasets

Please check the [datasets](https://github.com/mooselab/suppmaterial-PostDupGPT3/tree/master/datasets) folder for details.

## Installation

To use this project, you need to have the following dependencies installed:

- Python 3.7 or higher
- Some Python libraries (Specified in requirements.txt)

Clone this repository to your local machine:
```bash
git clone https://github.com/mooselab/suppmaterial-PostDupGPT3.git
```

You can install the Python libraries using pip:
```bash
cd ./DuplicatePostDetectionGPT3/src
pip install -r requirements.txt
```

## Model Training

This replication package contains a tiny sample dataset for testing the codes.

### Train with Triplet loss

```
cd ./src
python ./train_triplet_loss.py
```
### Train with MNR loss

```
cd src
python ./train_MNR_loss.py
```

### DupPredictor
We re-implemented the [DupPredictor](https://link.springer.com/article/10.1007/s11390-015-1576-4). The re-implementation is under `DupPredictor_ReImp` folder. 

A dataset use to do the grid search for parameters and [CQADupStack](https://github.com/D1Doris/CQADupStack) testsets of nine domains are included.

#### Usage:
1. Install the requirements:
```
cd ./DupPredictor_ReImp
pip install -r requirements.txt
```
2. Search for the best parameters for composer:
```
python ./DupPred_param_search.py
```
This process involves the training of the topic model. The trained model will be saved and used in the evaluation process.

3. Predict with testsets:
```
python DupPredictor.py
```
By running it, all the nine sub-domains will be iterated.


### Tools

The `Tools` folder contains codes used to generate the GPT-3 embeddings and print the prediction results.

### Citation

Xingfang Wu, Heng Li, Nobukazu Yoshioka, Hironori Washizaki, Foutse Khomh, Refining GPT-3 Embeddings with a Siamese Structure for Technical Post Duplicate Detection, Proceedings of the 31st IEEE International Conference on Software Analysis, Evolution, and Reengineering (SANER), March 12 - 15, 2024, Rovaniemi, Finland, IEEE.

### License

This project is licensed under the [MIT License](https://github.com/mooselab/DuplicatePostDetectionGPT3/blob/master/LICENSE).
