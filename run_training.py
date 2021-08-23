import torch

from dataset.AudioDataset import AudioDataset, AudioDatasetLazy
from models.models import AudioExpressionNet3

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger  
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb 

hyperparameter_defaults = dict(
    learning_rate=0.0001,
    batch_size = 512,
)

num_epochs = 30

wandb_logger = WandbLogger(project='AudioDrivenGeneration', log_model="all") 

wandb.init(config=hyperparameter_defaults)
config = wandb.config


data_path = "./data"
audio_dataset = AudioDataset(data_path, 8)


train_len = int(len(audio_dataset)*0.7)
test_len = len(audio_dataset) - train_len

train_set, test_set = torch.utils.data.random_split(audio_dataset, [train_len, test_len])


trainloader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size"], 
                                          shuffle=True, num_workers=8, pin_memory=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size=config["batch_size"], 
                                          shuffle=False, num_workers=8, pin_memory=True)

model = AudioExpressionNet3(8, learning_rate=config["learning_rate"])

lr_monitor = LearningRateMonitor(logging_interval='step')

trainer = pl.Trainer(gpus=1, logger=wandb_logger, log_every_n_steps=10, callbacks=[lr_monitor], max_epochs=num_epochs, default_root_dir="./checkpoints")
wandb_logger.watch(model)

trainer.fit(model, trainloader, testloader)