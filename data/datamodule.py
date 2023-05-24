import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .dataset import Era5TrainDataset
import os

class Era5DataModule(pl.LightningDataModule):
  def __init__(
    self,
    checkpoint_path=None,
    max_sampling_time_steps=1,
    batch_size=1, 
    train_crop_h=640,
    train_crop_w=1280,
  ):
    super().__init__()
    self.save_hyperparameters(logger=False)

  def setup(self, stage=None):
    self.train_ds = Era5TrainDataset(
      self.hparams.train_crop_h,
      self.hparams.train_crop_w,
      self.hparams.max_sampling_time_steps,
      checkpoint_path=self.hparams.checkpoint_path)

  def train_dataloader(self):
    pin_memory = True if torch.cuda.is_available() else False
    return DataLoader(
      self.train_ds,
      num_workers=1,
      batch_size=self.hparams.batch_size,
      pin_memory=pin_memory,
    )
