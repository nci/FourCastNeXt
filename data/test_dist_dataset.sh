set -eu
curr_path=$(realpath ../)
export PATH=$curr_path/../python_env/bin:$PATH
export PYTHONPATH=$curr_path

which python
which ray

set -eu

function cleanup {
  set +e
  ray stop -f
  set -e
}
trap "cleanup" EXIT

ray start --head --num-cpus=4 --disable-usage-stats --include-dashboard=False
python -u dataset.py
