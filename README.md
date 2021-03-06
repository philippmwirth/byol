# BYOL
PyTorch Implementation of the BYOL paper: [Bootstrap your own latent: A new approach to self-supervised Learning](https://arxiv.org/abs/2006.07733)


This is currently a work in progress. The code is a modified version of SimSiam [here](https://github.com/IgorSusmelj/simsiam-cifar10).

- Time per epoch is around 1 minute on a V100 GPU
- GPU usage is around 9 GBytes

**Todo:**

- [X] warmup learning rate from 0
- [X] report results on cifar-10
- [ ] create PR to add to lightly

### Installation
```
pip install -r requirements.txt
```

### Dependencies

- PyTorch
- PyTorch Lightning
- Torchvision
- lightly

### Benchmarks
We benchmark the BYOL model on the CIFAR-10 dataset following the KNN evaluation protocol.


Epochs | Batch Size | warmup | Test Accuracy | Peak GPU Usage
------:|---------:|------:|-------:|---------------:
200 | 512 | | 0.85 | 9.3GBytes
200 | 512 |&#x2611; | 0.86 | 9.3GBytes
800 | 512 |  | 0.91 | 9.3GBytes

Accuracy             |  Loss 
:-------------------------:|:-------------------------:
 |


### Paper

 [Bootstrap your own latent: A new approach to self-supervised Learning](https://arxiv.org/abs/2006.07733)
