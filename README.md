# A pytorch SimCLR Implementation

This is a pytorch implementation of [SimCLR](https://arxiv.org/abs/2002.05709).

## Requeriments:
- cudatoolkit=10.2
- pytorch=1.6.0
- torchvision=0.7.0
- opencv
- pandas
- pyyaml

## Training and evaluation

To train a model simply run:

```python
python cifar.py --config CONFIGFILE --gpu-id GPUID
```

where CONFIGFILE is the yaml configuration file. The yaml file for CIFAR is config-file.yaml.

To evaluate the model with a linear model run:

```python
python cifar_eval.py --config CONFIGFILE --gpu-id GPUID
```
