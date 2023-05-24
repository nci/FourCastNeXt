import os
import math
import json
import numpy as np
import torch
from datetime import datetime
from dateutil.relativedelta import relativedelta
from torch.utils.data import (
  IterableDataset,
  get_worker_info,
)
import torch.distributed as dist

from .era5 import get_training_data
from utils import get_logger

class Era5TrainDataset(IterableDataset):
  def __init__(self, crop_h, crop_w, max_sampling_time_steps, checkpoint_path=None):
    super(Era5TrainDataset).__init__()
    self.im_height = 721
    self.im_width = 1440

    self.crop_h = crop_h
    self.crop_w = crop_w
    self.max_sampling_time_steps = max_sampling_time_steps
    self.checkpoint_interval = 100
    self.checkpoint_path = checkpoint_path

  def __iter__(self):
    logger = get_logger(__name__)
    start_time = datetime(1959, 1, 1, hour=6)
    end_time = datetime(2017, 12, 31, hour=18)

    self.time_steps = []
    curr_time = start_time
    while curr_time <= end_time:
      self.time_steps.append(curr_time)
      curr_time += relativedelta(hours=6)

    if dist.is_available() and dist.is_initialized():
      rank = dist.get_rank()
      self.world_size = dist.get_world_size()
    else:
      rank = 0
      self.world_size = 1

    worker_info = get_worker_info()
    if worker_info is None:
      seed = rank
      self.proc_rank = rank
    else:
      seed = rank + worker_info.id
      self.proc_rank = rank + worker_info.id

    init_msg = f'rank={rank}, world_size={self.world_size}'
    if 'NODE_RANK' in os.environ:
      node_rank = os.environ['NODE_RANK']
      local_rank = os.environ['LOCAL_RANK']
      init_msg += f', node_rank={node_rank}, local_rank={local_rank}'
    init_msg += f', seed={seed}, worker_info={str(worker_info)}'

    logger.info(init_msg)

    self.rng = np.random.default_rng(seed)
    self.sample_idx = self.rng.choice(len(self.time_steps), len(self.time_steps), replace=False)
    self.s_idx = 0

    self.train_step = 0

    if self.checkpoint_path is not None and os.path.exists(self.checkpoint_path):
      if self.proc_rank == 0:
        logger.info(f'loading dataset checkpoint: {self.checkpoint_path}')
      with open(self.checkpoint_path, 'r') as f:
        ds_states = json.load(f)

      fast_forward_steps = ds_states['step']
      if self.proc_rank == 0:
        logger.info(f'fast-forwarding dataset {fast_forward_steps} steps')

      for s in range(fast_forward_steps):
        self._get_data(fast_forward=True)

    ## we must initialize ray from within dataloader's worker processes
    import ray
    ray.init(address='auto')
    self.ray = ray

    if self.proc_rank == 0:
      logger.info(ray.cluster_resources())

    self.obj_refs = []

    return self

  def get_max_tasks_per_rank(self):
    res = self.ray.cluster_resources()
    return math.ceil(float(res['CPU']) / self.world_size)

  def _get_data(self, fast_forward=False):
    idx = self.sample_idx[self.s_idx]

    query_time = self.time_steps[idx]

    if self.max_sampling_time_steps > 2:
      steps = self.rng.choice(self.max_sampling_time_steps, size=1, replace=False) + 1
      steps = np.sort(steps).tolist()
    else:
      steps = [self.max_sampling_time_steps,]

    start_x = self.rng.choice(self.im_width - self.crop_w + 1, replace=False)
    start_y = self.rng.choice(self.im_height - self.crop_h + 1, replace=False)

    if not fast_forward:
      data_ref = get_training_data.remote(query_time, steps, start_y, self.crop_h, start_x, self.crop_w)
    else:
      data_ref = None

    self.s_idx += 1
    if self.s_idx >= self.sample_idx.shape[0]:
      self.s_idx = 0
      self.sample_idx = self.rng.choice(len(self.time_steps), len(self.time_steps), replace=False)

    self.train_step += 1
    return data_ref

  def __next__(self):
    exce = None
    n_trials = 5
    for _ in range(n_trials):
      try:
        max_tasks = self.get_max_tasks_per_rank()
        while len(self.obj_refs) < max_tasks:
          data_ref = self._get_data()
          self.obj_refs.append(data_ref)

          if self.checkpoint_path is not None and self.proc_rank == 0 and self.train_step % self.checkpoint_interval == 0:
            ds_states = {'step': self.train_step}
            with open(self.checkpoint_path, 'w') as f:
              json.dump(ds_states, f)

        ready_refs, obj_refs = self.ray.wait(self.obj_refs, num_returns=1)
        self.obj_refs = obj_refs
        for data_ref in ready_refs:
          data = self.ray.get(data_ref)
          return {
            'input0': torch.from_numpy(data['input0']),
            'input1': torch.from_numpy(data['input1']),
            'targets': {s: torch.from_numpy(tgt) for s, tgt in data['targets'].items()},
          }

      except Exception as exc:
        exce = exc

    raise Exception(f'failed to get training data after {n_trials} attempts, exception: {exce}')

    
## tests
if __name__ == '__main__':
  from torch.utils.data import DataLoader
  import time

  train_ds = Era5TrainDataset(640, 1280, 600, checkpoint_path='./ds_states.json')


  data_loader = DataLoader(
          train_ds,
          num_workers=1,
          batch_size=1,
        )

  dl_iter = iter(data_loader)
  for ib, batch in enumerate(dl_iter):
    print(ib, batch.keys(), batch['targets'].keys())

    if ib >= 5:
      break

