import numpy as np
import math
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from datetime import datetime
from dateutil.relativedelta import relativedelta
import ray
import zarr
import argparse
from pprint import pprint
import os
import sys

num_channels = 20
image_height = 721
image_width = 1440

def get_prediction_times(start_datetime, end_datetime, prediction_length):
  prediction_times = []

  curr_time = start_datetime
  while curr_time <= end_datetime:
    pred_times = []
    for ip in range(prediction_length+1):
      pred_time = curr_time + relativedelta(hours=ip*6)
      pred_times.append(pred_time)

      if ip == prediction_length:
        next_time = pred_time

    prediction_times.append(pred_times)
    curr_time = next_time

  return prediction_times

def parse_datetime(dt_str):
  try:
    return datetime.strptime(dt_str, '%Y-%m-%dT%H')
  except Exception as exc:
    logger = get_logger(__name__)
    logger.error(f'invalid start time format, valid format is %Y-%m-%dT%H e.g. 2018-01-01T06')
    raise

def init_output_schema(output_path, num_channels, prediction_times, image_height, image_width):
  n_ics = len(prediction_times)
  assert n_ics > 0
  prediction_length = len(prediction_times[0]) - 1
  n_times = n_ics * prediction_length

  ds_root = zarr.open(args.output_path, 'w')
  ds_vars = []
  for ic in range(num_channels):
    var_name = channel_to_var(ic)
    ds_var = ds_root.create(var_name,
        shape=(n_times, image_height, image_width),
        chunks=(1, image_height, image_width),
        fill_value=np.nan,
        dtype=np.float32)
    ds_var.attrs['_ARRAY_DIMENSIONS'] = ['time', 'y', 'x']
    ds_vars.append(ds_var)

  coord_time = ds_root.create('time',
      shape=(n_times, ),
      chunks=(n_times, ),
      dtype='datetime64[ns]')
  coord_time.attrs['_ARRAY_DIMENSIONS'] = ['time', ]

  pred_times = []
  for pred_t in prediction_times:
    for t in pred_t[1:]:
      pred_times.append(np.datetime64(t))
  coord_time[:] = np.array(pred_times, dtype='datetime64[ns]')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--num-pred-steps', required=True, type=int)
  parser.add_argument('--checkpoint-path', required=True, type=str)
  parser.add_argument('--start-time', required=True, type=str)
  parser.add_argument('--end-time', default='', type=str)
  parser.add_argument('--output-path', default='', type=str)
  parser.add_argument('--num-data-workers', default=12, type=int)
  parser.add_argument('--skip-rmse', default=False, action='store_true')
  args = parser.parse_args()

  pprint(args)

  try:
    ray.init(address='auto')
  except Exception as exc:
    ray.init(address=None, num_cpus=args.num_data_workers)

  curr_dir = os.path.dirname(os.path.realpath(__file__)) 

  if curr_dir not in sys.path:
    sys.path.append(curr_dir)

  from model import FourCastNetModule
  from data.era5 import get_training_data, channel_to_var
  from utils import get_logger

  logger = get_logger(__name__)

  if len(args.end_time) == 0:
    args.end_time = args.start_time

  start_time = parse_datetime(args.start_time)
  end_time = parse_datetime(args.end_time)

  assert end_time >= start_time

  has_outputs = len(args.output_path) > 0
  calc_rmse = not args.skip_rmse
  assert has_outputs or calc_rmse

  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  model = FourCastNetModule.load_from_checkpoint(args.checkpoint_path, map_location=device).eval()

  precision = 32 ## we use f32 for inference
  model.hparams.precision = precision

  model.to(device)

  prediction_times = get_prediction_times(start_time, end_time, args.num_pred_steps)
  if has_outputs:
    init_output_schema(args.output_path, num_channels, prediction_times, image_height, image_width)

  n_ics = len(prediction_times)
  if calc_rmse:
    avg_rmses = np.zeros((n_ics,), dtype=np.float32)
    channel_rmses = np.zeros((n_ics, num_channels), dtype=np.float32)

  obj_refs = []
  obj_ref_info = {}
  num_tasks = 0
  max_tasks = math.ceil(args.num_data_workers * 1.2)
    
  from torch.profiler import profile, record_function, ProfilerActivity

  with torch.profiler.profile(
        # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/fcnx'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
  ) as prof:
        
      for ic in range(n_ics):
        query_time = prediction_times[ic][0]
        logger.info(f'initial condition({ic+1}/{n_ics}): {query_time}')

        obj = get_training_data.remote(query_time, [args.num_pred_steps, ], 0, image_height, 0, image_width,
          has_input1=False, has_target=calc_rmse)
        obj_refs.append(obj)
        obj_ref_info[obj] = ic

        while (len(obj_refs) >= max_tasks or ic == n_ics - 1) and num_tasks < n_ics:
          ready_refs, obj_refs = ray.wait(obj_refs, num_returns=1)
          for obj in ready_refs:
            icc = obj_ref_info[obj]
            del obj_ref_info[obj]

            input_batch = ray.get(obj)

            input_batch['input0'] = torch.from_numpy(input_batch['input0'])[None, ...].to(device)
            input_batch['n_pred_steps'] = args.num_pred_steps

            if calc_rmse:
              for step, target in input_batch['targets'].items():
                gt = torch.from_numpy(input_batch['targets'][step])
                break

            with torch.inference_mode():
              prediction = model.predict_step(input_batch, 0)

            if has_outputs:
              t_start = icc * args.num_pred_steps
              t_end = t_start + args.num_pred_steps
              pred_np = prediction.cpu().numpy()
              with zarr.open(args.output_path) as ds:
                for cc in range(num_channels):
                  var_name = channel_to_var(cc)
                  ds[var_name][t_start:t_end, ...] = pred_np[:, cc, ...]

            if calc_rmse:
              avg_rmses[icc] = torch.sqrt(F.mse_loss(prediction[-1, ...], gt)).item()
              for cc in range(num_channels):
                channel_rmses[icc, cc] = torch.sqrt(F.mse_loss(prediction[-1, cc, ...], gt[cc, ...])).item()

            num_tasks += 1

      if calc_rmse:
        for ic in range(n_ics):
          query_time = prediction_times[ic][0]
          avg_rmse = avg_rmses[ic]
          logger.info(f'{query_time}: average rmse: {avg_rmse:.6f}, steps: {args.num_pred_steps}')
          for cc in range(num_channels):
            rmse = channel_rmses[ic, cc]
            logger.info(f'{query_time}: channel: {cc}, rmse: {rmse:.6f}, steps: {args.num_pred_steps}')
