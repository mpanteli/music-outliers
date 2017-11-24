# music-outliers

Data and code for the journal publication:
Panteli, M., Benetos, E., Dixon, S. A computational study on outliers in world music. *under-review*.

Listen to music outliers in this [interactive demo](https://mpanteli.github.io/music-outliers/demo/outliers.html). More info about the demo [here](https://github.com/mpanteli/music-outliers/blob/master/demo/README.md). 

Extracted audio features and additional data uploaded [here](http://c4dm.eecs.qmul.ac.uk/worldmusicoutliers/). Copy these into the ['data' directory](https://github.com/mpanteli/music-outliers/blob/master/data/) to reproduce results.

## Overview

This project:
- extracts audio features for a set of world music recordings
- learns linear projections and 
- evaluates music similarity with a classification task predicting the country of origin of recordings 
- uses the (best) learned space to detect outlier recordings and 
- explores relationships of music dissimilarity between different geographical areas of the world. 

## Usage

#### Load dataset and extract features

Given a set of audio recordings and metadata including the country of origin (e.g., as in data/metadata.csv): 
- split the dataset into train, validation, and test sets,
- extract features (scale transform, pitch bihistogram, MFCC stats, chroma stats) for each recording, 
- standardise and concatenate to a numpy array of N_samples x F_features,  
- output each dataset as a pickle file. 

```python
python scripts/load_dataset.py
```

![alt tag](https://raw.githubusercontent.com/mpanteli/music-outliers/master/data/methodology.png)

#### Learn a feature space of music similarity 

Use supervised (LDA, SSNMF) and unsupervised (PCA, NMF) methods to learn features in a low-dimensional space. Learning is applied on 8-second frame-level descriptors and the output is averaged accross all frames of each recording. 

```python
python scripts/map_and_average.py
```

#### Classification

A classification task predicting the country of origin of a recording is used to assess the similarity in the learned feature spaces and optimise parameters. 

```python
python scripts/classification.py
```

The confusion matrix of 137 classes is exported into a .csv (see [confusion_matrix.csv](https://github.com/mpanteli/music-outliers/blob/master/data/confusion_matrix.csv) for the numerical data and [confusion_matrix_labels.csv](https://github.com/mpanteli/music-outliers/blob/master/data/confusion_matrix_labels.csv) for the class labels).  
See also the notebook [results_classification.ipynb](https://github.com/mpanteli/music-outliers/blob/master/notebooks/results_classification.ipynb).

#### Outliers

Given the best feature space (LDA projection) as estimated via the classification task, detect recordings that can be considered outliers with respect to the rest of the corpus. Outliers are estimated for each feature category (rhythm, melody, harmony, timbre), for the combination of all features, and for subsets of the dataset that define spatial country neighbourhoods. 

```python
python scripts/outliers.py
```

See also the notebook [results_outliers.ipynb](https://github.com/mpanteli/music-outliers/blob/master/notebooks/results_outliers.ipynb).

#### Hubness and sensitivity experiments

Effects of hubness resulting from processing high-dimensional vectors are measured in the notebook [hubness.ipynb](https://github.com/mpanteli/music-outliers/blob/master/notebooks/hubness.ipynb).

The stability of outlier results under different subsets of the original dataset is assessed in the notebook [sensitivity_experiment_outliers.ipynb](https://github.com/mpanteli/music-outliers/blob/master/notebooks/sensitivity_experiment_outliers.ipynb).
