
# About the datasets

## Introduction

This folder contains only a sample dataset, which contains GPT-3 embeddings, for testing the code. 
To access the entire dataset constructed by us and the [CQADupStack](https://github.com/D1Doris/CQADupStack) benchmark dataset adapted to our codes, kindly visit this link: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10067702.svg)](https://doi.org/10.5281/zenodo.10067702)

Our constructed dataset requires data conversion to match the format and structure of the sample dataset. GPT-3 embeddings need to be generated by users with our provided code. The benchmark dataset can be directly used to train and test the model, as we included the GPT-3 embeddings for all the posts.


## Details for the **sample** dataset

This sample dataset is intended for code testing purposes. It is a tiny subset of the entire dataset. It contains the GPT-3 embeddings for posts.
- **Format:** `.npy`

### How to load the sample dataset

```
import numpy as np
train_set = np.load('./tiny_train.npy', allow_pickle = True)[()]
test_set = np.load('./tiny_test.npy', allow_pickle = True)[()]
```

### The structure of the sample dataset

The posts are stored in `dict` data structure of Python.
```
train_set.keys()
# out: dict_keys(['ori', 'dup', 'rel'])
```
- `dup`: All duplicate posts (query posts).
- `ori`: All original posts (candidate posts).
- `rel`: A mapping from duplicate posts to their candidates.

#### Example
Posts:
```
train_set['dup'][44385544]
out: 
{'Id': '44385544',
 'PostTypeId': '1',
 'AcceptedAnswerId': '44385625',
 'CreationDate': '2017-06-06T08:53:50.603',
 'Score': '6',
 'ViewCount': '10427',
 'Body': "<p>Why is this not working ?..... approach.</p>\n",
 'OwnerUserId': '2612235',
 'LastEditorUserId': '2612235',
 'LastEditDate': '2017-06-06T08:59:49.333',
 'LastActivityDate': '2017-06-06T09:04:11.613',
 'Title': 'How to define a function inside a loop',
 'Tags': '<python><function-pointers>',
 'AnswerCount': '3',
 'CommentCount': '4',
 'ClosedDate': '2017-06-06T09:10:39.230',
 'ContentLicense': 'CC BY-SA 3.0',
 'gpt': array([-0.01197202,  0.01264388, ...,  0.00318956, -0.02457301])}
```

Duplicate relations:
```
train_set['rel'][45189775]
out: [44260491, 18195758]
```

## Details for the **benchmark** dataset

A subset of the [CQADupStack](https://github.com/D1Doris/CQADupStack). It contains the studied nine sub-domains. Each post comes with its GPT-3 embeddings. The data structure is the same as the sample dataset, except a `gpt` key is available for each post.

- **Format:** `.npz`

## Details for our **constructed** dataset

- **Format:** `.csv`

### Data Overview

The dataset contains all duplicate post pairs from Stack Overflow up to December 2022, with an 80%/20% split between the training and test sets. In total, 723,008 duplicate post pairs are included.

We do not provide GPT-3 embeddings for the entire dataset, as the data size would be huge. Please generate the embeddings with our provided code, which requires you to insert your own OpenAI key.

### About the usage

You need to convert the dataset to `dict` (the exact same structure as the sample dataset) in python and store as '.npy' format files to train and test the models.

## License

This project is licensed under the MIT License.


