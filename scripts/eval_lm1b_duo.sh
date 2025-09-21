#!/bin/bash
#SBATCH -J eval_mdlm                  # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=100000                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu               # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

checkpoint_path=/share/kuleshov/ssahoo/flow-ode/flow-ode-W2ZcFy-small-conjugate-lm1b-wrap-1M-gmin-350-gmax-175-3/checkpoints/7-100000.ckpt

export HYDRA_FULL_ERROR=1

srun python -u -m main \
  mode=ppl_eval \
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  data=lm1b-wrap \
  model=small \
  model.length=128 \
  algo=duo_base \
  eval.checkpoint_path=$checkpoint_path \
  sampling.num_sample_batches=0 \
  +wandb.offline=true