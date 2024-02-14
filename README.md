FourCastNeXt
============

Overview
--------

This repo contains scripts to perform FourCastNeXt training and inference using ERA5 from [NCI](https://nci.org.au/) project `rt52`.

For technical details in the model and training methods, please refer to the [preprint](https://arxiv.org/abs/2401.05584).

Citation
-------

```bibtex
@article{guo2024fourcastnext,
  title={FourCastNeXt: Improving FourCastNet Training with Limited Compute},
  author={Edison Guo and Maruf Ahmed and Yue Sun and Rahul Mahendru and Rui Yang and Harrison Cook and Tennessee Leeuwenburg and Ben Evans},
  journal={arXiv preprint arXiv:2401.05584},
  year={2024}
}
```

Setup
-----

* Ask to join NCI project rt52 on [mancini](https://my.nci.org.au/mancini).

* Run `bash setup.sh` to set up the environment. This script sets up a Python virtualenv with all the
  dependencies. The virtualenv directory `python_env` is in the same directory as `setup.sh`.

* The entrypoint of training is `run_trainer.pbs`. The inference script is `run_inference.pbs`.
  Before you run these scripts, please open them in an text editor and fill in `<your NCI project>`
  for `run_trainer.pbs`, and `<output path>` and `<checkpoint path>` for `run_inference.pbs`.

Training cluster
----------------

`run_trainer.pbs` sets up a training cluster. The training cluster consists of a GPU cluster for
Distributed Data Parallel (DDP) training and a ray cluster for data loading. The ray cluster uses the
current GPU node as the coordinator and launches three separate CPU Gadi jobs for the data workers.
The data workers will join the ray cluster as soon as the CPU Gadi jobs start. The data workers will
be automatically shut down when the ray coordinator is being shut down.
