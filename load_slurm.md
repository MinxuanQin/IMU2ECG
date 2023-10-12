# Instructions for euler & python environment setup

1. On euler, each time load module:

```
module load gcc/8.2.0 python_gpu/3.10.4 cuda/11.8.0 git-lfs/2.3.0 git/2.31.1 eth_proxy
```

2. Set up python environment

```
py_venv_dir = "${SCRATCH}/.python_venv"
python -m venv ${py_venv_dir}/imu2ecg --upgrade-deps
# install required packages
${SCRATCH}/.python_venv/imu2ecg/bin/pip install -r requirements.txt --cache-dir ${SCRATCH}/pip_cache

# activate
source "${SCRATCH}/.python_venv/imu2ecg/bin/activate"
```
