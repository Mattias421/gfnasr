#!/bin/bash
#SBATCH --time=80:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=results/slurm/%x-%a.out
#SBATCH --array=0

LR=(1e-1 1e-2 1e-3 1e-4 1e-5 2e-1 2e-2 2e-3 2e-5 2e-4) 
export WANDB_TAGS=lr_sweep
export CUDA_LAUNCH_BLOCKING=1

lr=${LR[$SLURM_ARRAY_TASK_ID]}

module load CUDA/12.4.0 

source ./.venv/bin/activate

python train.py --device=cpu hparams/train.yaml --trial_id=lr_$lr --evaluate=false --lr_whisper=$lr --number_of_epochs=500
#SBATCH --partition=dcs-gpu,gpu
#SBATCH --account=dcs-res
#SBATCH --gres=gpu:1
