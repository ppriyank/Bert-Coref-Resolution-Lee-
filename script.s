#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=GPUDemo
#SBATCH --mail-type=END
#SBATCH --mail-user=pp1953p@nyu.edu
#SBATCH --output=slurm_%j.out

module load anaconda3/4.3.1

conda activate pathak
module load cuda/9.0.176
cd /scratch/pp1953/codes/
python train.py best 



