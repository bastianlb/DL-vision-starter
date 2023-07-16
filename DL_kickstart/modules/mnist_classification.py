from torch import nn, optim
import torchmetrics as metrics
import pytorch_lightning as pl

# new pytorch lightning module for an mnist classification task
class MNISTClassification(pl.LightningModule):

    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.lr = config.train.learning_rate
        self.max_epochs = config.train.max_epochs
        # define the model loss, optimizer, and scheduler
        # these could also be configured with hyperparameters
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR
        # add an accuracy metric for each scenario
        self.train_acc = metrics.Accuracy('multiclass', num_classes=10)
        self.val_acc = metrics.Accuracy('multiclass', num_classes=10)
        self.test_acc = metrics.Accuracy('multiclass', num_classes=10)

    def forward(self, x):
        # forward pass through model
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # training step
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.train_acc(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def train_batch_end(self, outputs):
        self.log('train_acc', self.train_acc.compute())

    def validation_step(self, batch, batch_idx):
        # validation step
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.val_acc(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def on_validation_epoch_end(self):
        self.log('val_acc', self.val_acc.compute())

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.test_acc(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def on_test_epoch_end(self):
        self.log('test_acc', self.test_acc.compute())

    def configure_optimizers(self):
        # configure optimizers and schedulers for training
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        scheduler = self.scheduler(optimizer, self.max_epochs, verbose=True)
        return [optimizer], [scheduler]
