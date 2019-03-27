## A PyTorch implementation of GraphSAGE

This package contains a PyTorch implementation of [GraphSAGE](http://snap.stanford.edu/graphsage/).

#### Authors of this code package: [Tianwen Jiang](http://ir.hit.edu.cn/~twjiang/) (tjiang2@nd.edu), [Tong Zhao](tong-zhao.com) (tzhao2@nd.edu).




## Environment settings

- python==3.6.8
- pytorch==1.0.0




## Basic Usage

**Main Parameters:**

```
Input graph path. Defult is '../data/rating_train.dat' (--train-data)
Test dataset path. Default is '../data/rating_test.dat' (--test-data)
Name of model. Default is 'default' (--model-name)
Number of dimensions. Default is 128 (--d)
Number of negative samples. Default is 4 (--ns)
Size of window. Default is 5 (--ws)
Trade-off parameter $\alpha$. Default is 0.01 (--alpha)
Trade-off parameter $\beta$. Default is 0.01 (--beta)
Trade-off parameter $\gamma$. Default is 0.1 (--gamma)
Learning rate $\lambda$. Default is 0.01 (--lam)
Maximal iterations. Default is 50 (--max-iters)
Maximal walks per vertex. Default is 32 (--maxT)
Minimal walks per vertex. Default is 1 (--minT)
Walk stopping probability. Default is 0.15 (--p)
Calculate the recommendation metrics. Default is 0 (--rec)
Calculate the link prediction. Default is 0 (--lip)
File of training data for LR. Default is '../data/wiki/case_train.dat' (--case-train)
File of testing data for LR. Default is '../data/wiki/case_test.dat' (--case-test)
File of embedding vectors of U. Default is '../data/vectors_u.dat' (--vectors-u)
File of embedding vectors of V. Default is '../data/vectors_v.dat' (--vectors-v)
For large bipartite, 1 do not generate homogeneous graph file; 2 do not generate homogeneous graph. Default is 0 (--large)
```

**Usage**

We provide one processed dataset DBLP. It contains:

- A training dataset     ./data/rating_train.dat 
- A testing dataset      ./data/rating_test.dat
