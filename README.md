## A PyTorch implementation of GraphSAGE

This package contains a PyTorch implementation of [GraphSAGE](http://snap.stanford.edu/graphsage/).

#### Authors of this code package: [Tianwen Jiang](http://ir.hit.edu.cn/~twjiang/) (tjiang2@nd.edu), [Tong Zhao](tong-zhao.com) (tzhao2@nd.edu).




## Environment settings

- python==3.6.8
- pytorch==1.0.0




## Basic Usage

**Main Parameters:**

```
--dataSet The input graph dataset, default is cora which is provided in this repository.
--agg_func: The aggregate function to be used, default is the MEAN talked in paper.
Input graph path. Defult is '../data/rating_train.dat' (--train-data)
Test dataset path. Default is '../data/rating_test.dat' (--test-data)
Name of model. Default is 'default' (--model-name)
Number of dimensions. Default is 128 (--d)
Number of negative samples. Default is 4 (--ns)
Size of window. Default is 5 (--ws)
Trade-off parameter $\alpha$. Default is 0.01 (--alpha)
Trade-off parameter $\beta$. Default is 0.01 (--beta)
Trade-off parameter $\gamma$. Default is 0.1 (--gamma)
```

**Usage**

We provide one processed dataset DBLP. It contains:

- A training dataset     ./data/rating_train.dat 
- A testing dataset      ./data/rating_test.dat
