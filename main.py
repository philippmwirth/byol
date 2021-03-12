import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import pytorch_lightning as pl
import lightly
from lightly import data
from lightly import loss

from byol import BYOL
from utils import knn_predict, BenchmarkModule

num_workers = 8
max_epochs = 200
knn_k = 200
knn_t = 0.1
classes = 10
batch_size = 512
seed=1

pl.seed_everything(seed)

# use a GPU if available
gpus = 1 if torch.cuda.is_available() else 0
device = 'cuda' if gpus else 'cpu'

# Use SimCLR augmentations, additionally, disable blur
collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=32,
    gaussian_blur=0.,
)

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

dataset_train_ssl = lightly.data.LightlyDataset.from_torch_dataset(
    torchvision.datasets.CIFAR10(
        root='data',
        train=True,
        download=True))
dataset_train_kNN = lightly.data.LightlyDataset.from_torch_dataset(torchvision.datasets.CIFAR10(
    root='data',
    train=True,
    transform=test_transforms,
    download=True))
dataset_test = lightly.data.LightlyDataset.from_torch_dataset(torchvision.datasets.CIFAR10(
    root='data',
    train=False,
    transform=test_transforms,
    download=True))

dataloader_train_ssl = torch.utils.data.DataLoader(
    dataset_train_ssl,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=num_workers
)
dataloader_train_kNN = torch.utils.data.DataLoader(
    dataset_train_kNN,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)
dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)


class BYOLModule(BenchmarkModule):
    def __init__(self, dataloader_kNN, gpus, classes, knn_k, knn_t):
        super().__init__(dataloader_kNN, gpus, classes, knn_k, knn_t)
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        # create a simsiam model based on ResNet
        # note that bartontwins has the same architecture
        self.resnet_byol = BYOL(
            self.backbone,
            num_ftrs=512,
            hidden_dim=1024,
            out_dim=512,
            m=0.996
        )
        self.criterion = lightly.loss.SymNegCosineSimilarityLoss()
            
    def forward(self, x):
        self.resnet_simsiam(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        y0, y1 = self.resnet_byol(x0, x1)
        # symmetrize the outputs of byol and calculate the loss
        loss = self.criterion(y0, y1)
        self.log('train_loss_ssl', loss)
        return loss

    # learning rate warm-up
    def optimizer_steps(self,
                        epoch=None,
                        batch_idx=None,
                        optimizer=None,
                        optimizer_idx=None,
                        optimizer_closure=None,
                        on_tpu=None,
                        using_native_amp=None,
                        using_lbfgs=None):        

        # update params
        optimizer.step()
        optimizer.zero_grad()

    def configure_optimizers(self):
        lr = 2e-2 * batch_size / 256 # linear scaling of lr
        optim = torch.optim.SGD(self.resnet_byol.parameters(), lr=lr,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


model = BYOLModule(dataloader_train_kNN, gpus=gpus, classes=classes, knn_k=knn_k, knn_t=knn_t)

trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus)

trainer.fit(
    model,
    train_dataloader=dataloader_train_ssl,
    val_dataloaders=dataloader_test
)

print(f'Highest test accuracy: {model.max_accuracy:.4f}')
