import os
import numpy as np
import torch
import xarray as xr
import zarr
from datetime import datetime
from dateutil.relativedelta import relativedelta
import ray

## all variables: https://github.com/NVlabs/FourCastNet/blob/master/data_process/parallel_copy_small_set.py
def get_vars():
  sfc_vars = {'10u':0, '10v':1, '2t':2, 'sp':3, 'msl':4, 'tcwv':19}
  pl_vars = {
    't': ([850, 500], [5, 15, ]),
    'u': ([1000, 850, 500, ], [6, 9, 12, ]),
    'v': ([1000, 850, 500, ], [7, 10, 13, ]),
    'z': ([1000, 850, 500, 50], [8, 11, 14, 16, ]),
    'r': ([850, 500], [18, 17, ]),
  }
  return sfc_vars, pl_vars

def channel_to_var(channel_idx):
  sfc_vars, pl_vars = get_vars()

  for v, cdx in sfc_vars.items():
    if cdx == channel_idx:
      return v

  for v, info in pl_vars.items():
    levels, channels = info
    for idx, lvl in enumerate(levels):
      if channels[idx] == channel_idx:
        return f'{v}{lvl}'

def get_files(query_time, root_path='/g/data/rt52/era5'):
  year = query_time.year
  month = query_time.month

  start_date = datetime(year, month, 1)
  end_date = start_date + relativedelta(months=1) - relativedelta(days=1)

  start_ts = start_date.strftime('%Y%m%d')
  end_ts = end_date.strftime('%Y%m%d')
  ts = f'{start_ts}-{end_ts}'

  file_info = {}
  sfc_vars, pl_vars = get_vars()

  sfc_path = os.path.join(root_path, 'single-levels/reanalysis')
  for v, channel_idx in sfc_vars.items():
    file_info[v] = {
      'path': os.path.join(sfc_path, f'{v}/{year}/{v}_era5_oper_sfc_{ts}.nc'),
      'level': 0,
      'channel':channel_idx
    }

  pl_path = os.path.join(root_path, 'pressure-levels/reanalysis')
  for v, info in pl_vars.items():
    levels, channels = info
    for idx, lvl in enumerate(levels):
      v_name = f'{v}{lvl}'
      file_info[v_name] = {
        'path': os.path.join(pl_path, f'{v}/{year}/{v}_era5_oper_pl_{ts}.nc'),
        'level': lvl,
        'channel': channels[idx],
      }

  return file_info  

def get_data(query_time, y_start, crop_h, x_start, crop_w):
  file_info = get_files(query_time)
  num_channels = len(file_info)

  time_q = np.datetime64(query_time)

  fields = np.zeros((num_channels, crop_h, crop_w), dtype=np.float32)

  file_ds = {}
  for v, info in file_info.items():
    if info['path'] not in file_ds:
      ds = xr.open_dataset(info['path'])
      file_ds[info['path']] = ds
    else:
      ds = file_ds[info['path']] 

    ds = ds.sel(time=time_q, method='nearest')
    var_name = list(ds.keys())[0]

    level = info['level']
    channel = info['channel']

    if level == 0:
      fields[channel, ...] =  ds[var_name][y_start:y_start+crop_h, x_start:x_start+crop_w].values
    else:
      ds = ds.sel(level=level)
      fields[channel, ...] =  ds[var_name][y_start:y_start+crop_h, x_start:x_start+crop_w].values

  for _, ds in file_ds.items():
    ds.close()

  return fields


def get_input_data(query_time, y_start, crop_h, x_start, crop_w):
  input_q = get_data(query_time, y_start, crop_h, x_start, crop_w)

  prev_time = query_time - relativedelta(hours=6) 
  prev_input = get_data(prev_time, y_start, crop_h, x_start, crop_w)
  return np.vstack([prev_input, input_q])

@ray.remote(num_cpus=1, scheduling_strategy='SPREAD', max_retries=3)
def get_training_data(query_time, steps, y_start, crop_h, x_start, crop_w, has_input1=True, has_target=True):
  assert len(steps) == 1
  input_data0 = get_input_data(query_time, y_start, crop_h, x_start, crop_w)

  targets = {}
  for s in steps:
    step_time = query_time + relativedelta(hours=s * 6) 
    if has_target:
      targets[s] = get_data(step_time, y_start, crop_h, x_start, crop_w)

    if has_input1:
      if s > 1:
        input1_time = step_time - relativedelta(hours=6) 
        input_data1 = get_input_data(input1_time, y_start, crop_h, x_start, crop_w)
      else:
        input_data1 = input_data0.copy()

  output = {
    'input0': input_data0,
  }
  if has_input1:
    output['input1'] = input_data1

  if has_target:
    output['targets'] = targets

  return output


## test
if __name__ == '__main__':
  query_times = [
    datetime(2020, 1, 1, hour=0),
    datetime(2020, 1, 1, hour=6),
  ]

  #get_training_data(datetime(2020, 1, 1, hour=0), 2, 2, 224, 2, 224)

  import time
  t0 = time.time()
  get_training_data(datetime(2022, 1, 1, hour=0), [7, 11, 8, 1, 20], 0, 640, 0, 1280)
  print(time.time() - t0)

  ##dst_file_path = 'test_outputs/gt.zarr'

  ##format_FourCastNet(query_times, dst_file_path)
