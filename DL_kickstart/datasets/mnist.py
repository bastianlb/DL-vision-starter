from torch.utils.data import DataLoader

import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision import transforms

# dataloader for moving mnist

class MNISTDatamodule(pl.LightningModule):

    def __init__(self, data_root, batch_size=32, img_size=28, num_workers=10):
        super().__init__()
        self.data_dir = data_root
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        val_transforms = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5])])

        if stage == "fit":
            tr_transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.RandomCrop(self.img_size, padding=2),
                transforms.Normalize([0.5], [0.5])
            ])
            self.mnist_train = MNIST(self.data_dir, train=True, transform=tr_transform)
            self.mnist_val = MNIST(self.data_dir, train=False, transform=val_transforms)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=val_transforms)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=val_transforms)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size,
                          num_workers=self.num_workers)
