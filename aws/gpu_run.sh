#!/bin/bash

## job name
#SBATCH --job-name=512
## filename for job standard output (stdout)
## %j is the job id, %u is the user id
#SBATCH --output=/fsx/ganayu/experiments/trial/sample-%j.out
#SBATCH --error=/fsx/ganayu/experiments/trial/sample-%j.err

## partition name
#SBATCH --partition=a100
## number of nodes
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
## time
#SBATCH --time 10000

## number of tasks per node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10

# Start clean
module purge

# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID

srun --label /fsx/ganayu/code/SuperShaper/aws/commands.sh