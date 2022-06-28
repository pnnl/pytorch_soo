#!/bin/bash
#SBATCH --job-name=sso_priming_expts_fr
#SBATCH --mail-type=ALL
#SBATCH --mail-user=esilk16@uw.edu
#SBATCH --ntasks=4 
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00 
#SBATCH --output=/people/chia402/Documents/sso_expts/logs/array%A-%a.log
#SBATCH --account=homotopy
#SBATCH --partition=tonga
#SBATCH --array=0-15 

pwd; hostname; date
echo $(nproc)
module load cuda/11.1
module load python/miniconda3.9
source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh
echo Sourced conda
conda activate /people/chia402/.conda/envs/ml20_tc
echo Activated ml20_env
which python3
echo This is task $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_MAX
python3 sso_priming_expts_pnnl.py --task_id $SLURM_ARRAY_TASK_ID --total_tasks $((SLURM_ARRAY_TASK_MAX + 1)) --optimizer fr
