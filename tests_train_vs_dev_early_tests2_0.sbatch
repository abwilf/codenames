#!/bin/bash
#SBATCH -p gpu_highmem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem 35GB 
#SBATCH --time 2-23:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dummyblah123@gmail.com # TODO
#SBATCH --chdir=/work/awilf/siqa
#SBATCH --output=/work/awilf/siqa/logs/%j.out # TODO
#SBATCH --error=/work/awilf/siqa/logs/%j.err # TODO

singularity exec -B /work/awilf/ --nv /work/awilf/siqa/gpt.sif \
wandb agent socialiq/siqa/kyuvmn9g