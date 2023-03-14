#!/bin/bash
#SBATCH --job-name=semembed # Job name
#SBATCH --nodes=1 # Run all processes on a single node
#SBATCH --ntasks=1 # Run a single task
#SBATCH --mem=1gb # Job memory request
#SBATCH --cpus-per-task=4 # Number of CPU cores per task
#SBATCH --gpus-per-node=2 # Number of GPU
#SBATCH --partition=gpupart # Time limit hrs:min:sec
#SBATCH --output=/home/du0/20CS30043/iit/IR/semantic_embed/IR-project/semantic-embeddings/out.log # Standard output and error log

source activate env
cd /home/du0/20CS30043/iit/IR/semantic_embed/IR-project/semantic-embeddings
make class_embed
make sem_embed
make evaluate
