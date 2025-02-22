#!/usr/bin/env bash
# Input python command to be submitted as a job
#SBATCH --output=logs/run_script-%j.out
#SBATCH --job-name run_script
#SBATCH --partition psych_day,psych_scavenge,psych_week,day,week,interactive
#SBATCH --time=6:00:00
#SBATCH --mem=250G

module load AFNI ; module load FSL ; source /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.3-centos7_64/etc/fslconf/fsl.sh ; module load dcm2niix ; . /gpfs/milgram/apps/hpc.rhel7/software/Python/Anaconda3/etc/profile.d/conda.sh ; conda activate /gpfs/milgram/project/turk-browne/kp578/conda_envs/harmonic

script=$1

python -u $script $2 $3 $4

