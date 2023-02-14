#!/bin/bash
#SBATCH -p gpu_long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem 25GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your_email>@gmail.com # TODO
#SBATCH --chdir=<your_current_dir>
#SBATCH --output=<your_current_dir>/logs/%j.out
#SBATCH --error=<your_current_dir>/logs/%j.err

mkdir -p logs

singularity exec -B /work/awilf/ --nv /work/awilf/AANG/aang.sif \
wandb agent socialiq/siqa/vpsu5te1