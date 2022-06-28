#!/bin/bash
#SBATCH --job-name=generate_nlcg_pkl
#SBATCH --mail-type=ALL
#SBATCH --mail-user=esilk16@uw.edu
#SBATCH --ntasks=4 
#SBATCH --time=1:00:00 
#SBATCH --account=homotopy
#SBATCH --partition=tonga

pwd; hostname; date
echo $(nproc)
module load cuda/11.1
module load python/miniconda3.9
source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh
echo Sourced conda
conda activate /people/silk112/.conda/envs/ml20_env
echo Activated ml20_env
which python3
python3 nlcg_to_pkl.py
