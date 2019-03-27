## A PyTorch implementation of GraphSAGE

This package contains a PyTorch implementation of [GraphSAGE](http://snap.stanford.edu/graphsage/).

#### Authors of this code package: [Tianwen Jiang](http://ir.hit.edu.cn/~twjiang/) (tjiang2@nd.edu), [Tong Zhao](tong-zhao.com) (tzhao2@nd.edu).




## Environment settings

- python==3.6.8
- pytorch==1.0.0




## Basic Usage

**Main Parameters:**

```
--dataSet     The input graph dataset. (default: cora)
--agg_func    The aggregate function. (default: Mean aggregater)
--epochs      Number of epochs. (default: 200)
--b_sz        Batch size. (default: 20)
--seed        Random seed. (default: 824)
--num_neg     Number of negative samples in each batch. (default: 100)
--config      Config file. (default: ./src/experiments.conf)
--cuda        Use cuda if declared.
```

**Usage**


