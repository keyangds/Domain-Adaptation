# Domain Adaptation on Spatial Temporal Data

## Overview

This repository contains the implementaion of several domain adaptation algorithms: <br>
[coral](https://arxiv.org/abs/1607.01719) <br>
[cotmix](https://arxiv.org/abs/2212.01555) <br>
[dann](https://arxiv.org/abs/1505.07818) <br>
[cdan](https://arxiv.org/abs/1705.10667) <br>
[dirt](https://arxiv.org/abs/1802.08735) <br>
[AdvSKM](https://www.ijcai.org/proceedings/2021/0378.pdf) <br>

and we are working on adding more...

## How to run the code 
The running command consists of following parameters: <br>
```--config```: Deciding the encoder framework and its hyperparameters set up. (Currently only tcn supports all da_methods) <br>
```--da_method```: Choosing the domain adaptation algorithm <br>
```--training```: Turning this as False can directly test based on pretrained model. <br>
```--fixed_mask```: Setting it as True to apply a default mask in training. <br>

Example 1: To test on air dataset using coral: <br>
```python main.py --config config/tcn/air.yaml --fixed_mask True --da_method 'coral'``` 

Example 2: To test on water dataset using dirt: <br>
```python main.py --config config/tcn/discharge.yaml --fixed_mask True --da_method 'dirt'``` 
