#!/usr/bin/bash

#SBATCH --job-name="imu2ecg"
#SBATCH --output=../logs/%j.out
#SBATCH --time=8:00:00

#SBATCH --nodes=1

#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus-per-node=rtx_2080_ti:1

# Load modules or your own conda environment here
source /cluster/apps/local/env2lmod.sh
module load gcc/8.2.0 python_gpu/3.10.4 cuda/11.8.0 git-lfs/2.3.0 git/2.31.1 eth_proxy

# update environment !!! this is important
source "${SCRATCH}/.python_venv/imu2ecg/bin/activate"
python sample.py
