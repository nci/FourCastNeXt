#!/bin/bash
#PBS -P <your NCI project>
#PBS -q gpuvolta
#PBS -l walltime=02:00:00
#PBS -l mem=95GB
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l jobfs=200GB
#PBS -l storage=gdata/rt52+gdata/<your NCI project>
#PBS -N pbs_fourcastnext_inference

set -eu

start_time='2018-01-01T00'
end_time='2018-01-02T00'
steps=3

output_path=<output path>
checkpoint_path=<checkpoint path>

module load cuda/11.7.0

curr_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
curr_path=${PBS_O_WORKDIR:-$curr_dir}

export PATH=$curr_path/python_env/bin:$PATH

(
cd $curr_path

python -u inference.py \
  --num-pred-steps=$steps \
  --checkpoint-path=$checkpoint_path \
  --output-path=$output_path \
  --start-time=$start_time \
  --end-time=$end_time
)
