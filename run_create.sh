#!/bin/bash -l
#SBATCH --job-name=dreamer
#SBATCH --output=/scratch/users/%u/dreamer-mcx/dreamer.out
#SBATCH --error=/scratch/users/%u/dreamer-mcx/dreamer.err
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-48:00
#SBATCH --chdir=/scratch/users/${USER}/dreamer-mcx
#SBATCH --nodes=1
#SBATCH --mem=64000
module load cuda
cd /users/$USER/dreamer-mcx-jax
source venv/bin/activate
cd dreamerv3
bash entrypoint.sh python dreamerv3/main.py --configs dmc_vision debug --method test --logdir logdir
