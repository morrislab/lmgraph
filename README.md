DESCRIPTION: This is the python code accompanying the paper:
```
Shankar Vembu, Quaid Morris. An Efficient Algorithm to Integrate Network 
and Attribute Data for Gene Function Prediction. In Proceedings of the 
Pacific Symposium on Biocomputing, 2015.
```

DEPENDENCIES:
python, numpy, scipy, scikit-learn

USAGE:
python lprop.py 'example' 24
python lmgraph.py 'example' 24

The code loads networks from /data/example_X.npy and the specific target label index. It splits the data into training, validation and test sets in the ratio of 3:1:1. It uses the validation set to tune the regularization parameter and computes the average AUC using the test set splits.

A typical output looks like:
```
python lprop.py 'example' 24
Mean AUC: 0.942680957275
python lmgraph.py 'example' 24
Mean AUC: 0.954113443939
```

The code has been tested with:
- python 2.7.10
- numpy 1.8.0
- scipy 0.13.0
- sklearn 0.17.1
