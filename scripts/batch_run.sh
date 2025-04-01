#!/bin/bash
#SBATCH --array=1-19                    # 20 tasks (1 to 20)
#SBATCH --ntasks=1                     # One task per job
#SBATCH --cpus-per-task=1              # One CPU per task
#SBATCH --output=log_%A_%a.out         # Output log file for each job
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=lisaleemcb@gmail.com
#
# Define the range of numbers for each task (each task gets a different range)
start=$(( ($SLURM_ARRAY_TASK_ID - 1) * 500 ))  # Start value for each job
end=$(( $start + 500 ))                      # End value for each job

# file name
sims="sims_a${start}de${end}.npy"

# use the bash shell
set -x
# echo each command to standard out before running it
date

# source bash profile
source /home/emc-brid/.bashrc
source ~/venvs/alfred/bin/activate

# run the Unix 'date' command
echo "Running with sims contained in file ${sims}"

# Run the program with the generated input file
python -u /home/emc-brid/alfred/scripts/make_kSZ.py --sims /home/emc-brid/${sims} --save_dir nells30_xion97

