#!/bin/csh
#SBATCH -p gpu,all
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --array=0-19%20



source /home/projects2/tianxu/miniconda3/etc/profile.d/conda.csh
conda activate tcr_gpu

set PYTHON = "/home/projects2/tianxu/miniconda3/envs/tcr_gpu/bin/python3"

set t_array = "0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4"
set v_array = "1 2 3 4 0 2 3 4 0 1 3 4 0 1 2 4 0 1 2 3"

set t = `echo $t_array | awk -v i=$SLURM_ARRAY_TASK_ID '{print $(i+1)}'`
set v = `echo $v_array | awk -v i=$SLURM_ARRAY_TASK_ID '{print $(i+1)}'`

#$PYTHON s02_train.py -c $1 -t $t -v $v > $1.$t.$v.out 2>&1
set log_file = "${1}.${t}.${v}.out"
$PYTHON train.py -c $1 -t $t -v $v >&! $log_file

scontrol show job $SLURM_JOB_ID
