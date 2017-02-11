DESCRIPTION: This is the python code accompanying the paper:
```
Shankar Vembu, Quaid Morris. An Efficient Algorithm to Integrate Network 
and Attribute Data for Gene Function Prediction. In Proceedings of the 
Pacific Symposium on Biocomputing, 2015.
```

DEPENDENCIES:
python, numpy, scipy

USAGE:

python lprop.py 'example' 24

python lmgraph.py 'example' 24

The first command runs the label propagation alorithm (baseline). The second command runs the LMGraph algorithm described in the paper.

The first argument ('example') is used to specify the name of the data files -- functional interaction networks and GO biological process function categories, i.e., target labels. The data files corresponding to 'example' has 7 networks each of them consists of 19,559 nodes (genes), and the number of function categories is 1,572. The second argument is to specify the target label index, so in the above example the labels in 25th function category (from 1,572 categories) is used as the target.

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
