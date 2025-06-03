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
#SBATCH --array=5-20

ARRAY=(1e-5 1e-6 1e-7 1e-8 1e-9) 
export WANDB_TAGS=reward_weight_sweep
export WANDB_SWEEP_ID=9s4olslm

val=${ARRAY[$SLURM_ARRAY_TASK_ID]}
val=$SLURM_ARRAY_TASK_ID

module load CUDA/12.4.0 

source ./.venv/bin/activate

python train.py hparams/train.yaml --trial_id=lr_$val --evaluate=false --reward_weight=$val --number_of_epochs=20 
