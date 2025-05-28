#!/bin/bash
#SBATCH --partition=dcs-gpu,gpu
#SBATCH --account=dcs-res
#SBATCH --gres=gpu:1
#SBATCH --time=80:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=results/slurm/%x-%a.out
#SBATCH --array=1

module load CUDA/12.4.0 

source ./.venv/bin/activate

python train.py hparams/train.yaml --trial_id=alpha_bs4 --batch_size=4 --number_of_epochs=100
