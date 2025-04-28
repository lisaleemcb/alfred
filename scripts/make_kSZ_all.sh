#!/bin/bash

#SBATCH --job-name=run_kSZ
#SBATCH --time=02-00:00:00
#SBATCH --ntasks=1                     # One task per job
#SBATCH --cpus-per-task=1              # One CPU per task
#SBATCH --output=output.log
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=lisaleemcb@gmail.com

# use the bash shell
set -x
# echo each command to standard out before running it
date

# source bash profile
source /home/emc-brid/.bashrc
source ~/venvs/alfred/bin/activate

echo "LIBRARY PATH IS"
env | grep LD_LIBRARY_PATH

# run the Unix 'date' command
echo "Hello world, from the Cluster!"
# run the Unix 'echo' command
# which mamba
# mamba activate kSZ
which python
python -u /home/emc-brid/alfred/scripts/make_kSZ.py --sims /home/emc-brid/sims_all.npy --save_dir nells30_v3.1 --n all
