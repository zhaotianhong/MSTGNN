# MSTGNN
This is the Pytorch implemention of GAMCN in the following paper: Multi-view spatio-temporal deep graph neural network model for travel demand prediction.

## Data
The real-world multi-modal transportation travel demand data from Shenzhen, china.
The collect travel demand data of bus, metro and taxi from 2018-12-01 to 2018-12-31.
You can download via the cloud drive: [Baidu](https://www.baidu.com/).
Graph structure data: including spatial adjacency graphs and OD graphs of different transport modes can be downloaded [Baidu](https://www.baidu.com/)



## Model

<p align="center">
  <img src="https://github.com/zhaotianhong/MSTGNN/blob/main/framework.png" width="400px">
</p>

## Requirements
Python 3.7  <br/>
Pytorch <br/>
scikit-learn <br/>
pandas <br/>
Numpy

## Model training and testing

You can modify the parameters in the `config.py` file, such as: prediction time step, learning rate, minimum batch, etc.

- For train

```python
python main.py gpu_id True model_name
```

- For test

```python
python main.py gpu_id False model_name
```


## Citation
A paper is in progress, and the citation will be added here.
