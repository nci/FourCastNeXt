import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

import os
import sys
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--max-train-steps', required=True, type=int)
  parser.add_argument('--max-sampling-time-steps', default=1, type=int)
  parser.add_argument('--base-lr', default=3*1e-3, type=float)
  parser.add_argument('--resume-checkpoint-path', default='', type=str)
  parser.add_argument('--best-model-path', default='', type=str)
  args = parser.parse_args()

  curr_dir = os.path.dirname(os.path.realpath(__file__)) 

  if curr_dir not in sys.path:
    sys.path.append(curr_dir)

  from model import FourCastNetModule
  from data import Era5DataModule
  from utils import get_logger

  logger = get_logger(__name__)

  base_lr = args.base_lr
  max_steps = args.max_train_steps
  batch_size = 1

  train_crop_h = 640
  train_crop_w = 1280

  checkpoint_every_n_train_steps = 500
  train_log_every_n_steps = min(max(max_steps * 0.05, 1), 100)
  trainer_root_dir = os.path.dirname(curr_dir)
  dataset_checkpoint_path = os.path.join(curr_dir, 'dataset_states.json')

  if len(args.best_model_path) == 0:
    best_model_path = os.path.join(curr_dir, 'best_model.txt')
  else:
    parent_dir = os.path.dirname(os.path.realpath(args.best_model_path))
    assert os.path.exists(parent_dir)
    best_model_path = args.best_model_path

  if len(args.resume_checkpoint_path) == 0:
    resume_checkpoint_path = None
  else:
    assert os.path.exists(args.resume_checkpoint_path)
    resume_checkpoint_path = args.resume_checkpoint_path

  if args.max_sampling_time_steps > 1:
    assert resume_checkpoint_path is not None

  pl.seed_everything(0)
  precision = 16 if torch.cuda.is_available() else 32

  means_np = np.load(f'{curr_dir}/stats/global_means.npy')[:, :-1]
  stds_np = np.load(f'{curr_dir}/stats/global_stds.npy')[:, :-1]

  means = torch.from_numpy(means_np).to(dtype=torch.float)
  stds = torch.from_numpy(stds_np).to(dtype=torch.float)

  if args.max_sampling_time_steps == 1:
    grad_accum_schedule={0:1, int(max_steps*0.3):2}
  else:
    grad_accum_schedule={0:2, }

  model = FourCastNetModule(
    means,
    stds,
    base_lr=base_lr,
    grad_accum_schedule=grad_accum_schedule,
    spatial_size=(train_crop_h, train_crop_w),
    precision=precision,
  )

  if args.max_sampling_time_steps > 1:
    logger.info(f'loading checkpoint for fine-tuning: {resume_checkpoint_path}')
    checkpoint = FourCastNetModule.load_from_checkpoint(resume_checkpoint_path)
    model.net.load_state_dict(checkpoint.net.state_dict())

  data_loader = Era5DataModule(
    max_sampling_time_steps=args.max_sampling_time_steps,
    checkpoint_path = dataset_checkpoint_path,
    batch_size=batch_size,
    train_crop_h=train_crop_h,
    train_crop_w=train_crop_w,
  )

  checkpoint_callback = pl.callbacks.ModelCheckpoint(
      every_n_train_steps=checkpoint_every_n_train_steps,
      verbose=True,
      monitor='step',
      mode='max',
      save_top_k=5,
      filename='model-{step}')

  strategy = DDPStrategy(find_unused_parameters=False)
  trainer = pl.Trainer(default_root_dir=trainer_root_dir,
      max_steps=max_steps,
      devices='auto',
      accelerator='auto',
      strategy=strategy,
      gradient_clip_val=1.0,
      precision=precision,
      log_every_n_steps=train_log_every_n_steps,
      enable_progress_bar=False,
      callbacks=[checkpoint_callback, ])

  if trainer.is_global_zero:
    logger.info(model)

  if args.max_sampling_time_steps > 1:
    resume_checkpoint_path = None
  trainer.fit(model, data_loader, ckpt_path=resume_checkpoint_path)

  if trainer.is_global_zero:
    with open(os.path.join(best_model_path), 'w') as f:
      f.write(checkpoint_callback.best_model_path)

    logger.info(f'best model path: {checkpoint_callback.best_model_path}')
