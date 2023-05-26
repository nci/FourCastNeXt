import os
import numpy as np
import xarray as xr
import zarr
from datetime import datetime
from dateutil.relativedelta import relativedelta

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

def get_data(query_time):
  file_info = get_files(query_time)
  num_channels = len(file_info)

  time_q = np.datetime64(query_time)

  fields = None
  for v, info in file_info.items():
    if info['level'] != 0:
      continue
    ds = xr.open_dataset(info['path'])
    var_name = list(ds.keys())[0]
    _, H, W = ds[var_name].shape

    ds.close()
    fields = np.zeros((1, num_channels, H, W), dtype=np.float32)
    break

  assert fields is not None, f'files not found for {query_time}'

  for v, info in file_info.items():
    ds = xr.open_dataset(info['path'])
    ds = ds.sel(time=time_q, method='nearest')
    var_name = list(ds.keys())[0]

    level = info['level']
    channel = info['channel']

    if level == 0:
      fields[0, channel, ...] =  ds[var_name].values
    else:
      ds = ds.sel(level=level)
      fields[0, channel, ...] =  ds[var_name].values

    ds.close()

  return fields


def format_FourCastNet(query_times, dst_path):
  assert isinstance(query_times, list), 'query_times must be a list'

  num_channels = 20
  img_shape_x = 720
  img_shape_y = 1440

  n_times = len(query_times)

  ds_root = zarr.open(dst_path, 'w')

  ds_vars = []
  for ic in range(num_channels):
    var_name = channel_to_var(ic)
    ds_var = ds_root.create(var_name,
        shape=(n_times, img_shape_x, img_shape_y),
        chunks=(1, img_shape_x, img_shape_y),
        fill_value=np.nan,
        dtype=np.float32)
    ds_var.attrs['_ARRAY_DIMENSIONS'] = ['time', 'x', 'y']
    ds_vars.append(ds_var)

  coord_time = ds_root.create('time',
    shape=(n_times, ),
    chunks=(n_times, ),
    dtype='datetime64[ns]')
  coord_time.attrs['_ARRAY_DIMENSIONS'] = ['time', ]

  time_q = [np.datetime64(qt) for qt in query_times]
  coord_time[:] = np.array(time_q, dtype='datetime64[ns]')

  for iqt, qt in enumerate(query_times):
    gt_data = get_data(qt).squeeze(0)
    for c_idx in range(num_channels):
      ds_vars[c_idx][iqt, ...] = gt_data[c_idx, :img_shape_x, ...]

## test
if __name__ == '__main__':
  query_times = [
    datetime(2018, 1, 1, hour=0),
    datetime(2018, 1, 8, hour=6),
  ]

  dst_file_path = 'test_outputs/gt.zarr'

  format_FourCastNet(query_times, dst_file_path)
