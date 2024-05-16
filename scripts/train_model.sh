#!/bin/bash
#SBATCH --job-name=mrnn
#SBATCH --nodes=1  
#SBATCH --ntasks=1    
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00  
#SBATCH --mail-type=fail  
#SBATCH --mail-user=tt1131@princeton.edu
#SBATCH --output=/scratch/gpfs/tt1131/projects/dp_tutorial/scripts/slurm-%j.out

module purge
module load anaconda3/2023.9
conda activate /scratch/gpfs/tt1131/.conda/envs/deepphase

wandb offline

python /scratch/gpfs/tt1131/projects/dp_tutorial/scripts/dp_train.py --proj_name dp_test_00