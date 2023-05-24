set -eu
curr_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

PYTHON=$(which python3)

$PYTHON -c 'import sys; assert sys.version_info.major >= 3 and sys.version_info.minor >= 7, f"requires Python 3.7+, current: {sys.version_info}"'

env_root=$(pwd)/python_env

mkdir -p $env_root
$PYTHON -m venv $env_root

(
  source $env_root/bin/activate
  pip install torch==1.13.1 torchvision==0.14.1 \
      --extra-index-url https://download.pytorch.org/whl/cu117
  pip install -r $curr_dir/requirements.txt 
)
