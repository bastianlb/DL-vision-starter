import hydra
import torch as ch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from DL_kickstart.modules.mnist_classification import MNISTClassification
from DL_kickstart.datasets.mnist import MNISTDatamodule
from DL_kickstart.models.mnist_transformer import VisionTransformer


# we use hydra as a config management system. This allows us to
# define a config file that can be used to define all the hyperparameters of our model.
# Many other tools are also available, such as yacs, or sacred.
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    # make sure to seed for reproducibility!
    pl.seed_everything(42)

    # use wandb logger. Tensorboard logger is also available
    logger = WandbLogger(
        name='neural-gen',
        project='neural-gen',
        log_model=True,
        save_dir='wandb',
    )

    # use mps if available, else use cuda with ddp
    accelerator = None
    if ch.backends.mps.is_available():
        accelerator = 'mps'
    else:
        accelerator = 'cuda'

    # define your data module. This should be a lightning module that
    # defines the train, val, and test dataloaders
    data_loader = MNISTDatamodule(data_root=cfg.data_root, batch_size=cfg.train.batch_size,
                                  img_size=cfg.mnist_transformer.img_size, num_workers=cfg.train.num_workers)

    # define a model to be trained on the data_loader defined above
    model = VisionTransformer(cfg.mnist_transformer)

    # define your main lightning module that implements
    # optimization logic including the train, val, and test/predict steps
    module = MNISTClassification(model, config=cfg)

    # callbacks can be a handy way to do things like logging, checkpointing, etc.
    # they provide general hooks that can be used across different training modules
    lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = [lr_monitor]

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        log_every_n_steps=10, accelerator=accelerator,
        logger=logger,
        deterministic=True,
        callbacks=callbacks,
        # using mixed precision can increases training speed, and reduce memory usage
        precision=16,
    )

    trainer.fit(module, datamodule=data_loader)
    trainer.test(module, datamodule=data_loader)


if __name__ == "__main__":
    main()

