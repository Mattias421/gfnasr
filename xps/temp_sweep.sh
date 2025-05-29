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
#SBATCH --array=0-5

TEMP=(16 32 64 128 256 512)
export WANDB_TAGS=bs_sweep
#export WANDB_SWEEP_ID=j5c5vvnf
#export CUDA_LAUNCH_BLOCKING=1


value=${TEMP[$SLURM_ARRAY_TASK_ID]}

module load CUDA/12.4.0 

source ./.venv/bin/activate

python train.py hparams/train.yaml --trial_id=bs_$value --batch_size=$value --evaluate=false --number_of_epochs=500
