#!/bin/bash -i
#SBATCH --partition psych_day,psych_scavenge,psych_week,day,week,interactive
#SBATCH --nodes 1
#SBATCH --time 24:00:00
#SBATCH --mem-per-cpu 100G
#SBATCH --job-name jup_ha
#SBATCH --output logs/jupyter-log-%J.txt
#SBATCH --mail-type ALL
#SBATCH --mail-user=kp578

# setup the environment

module load AFNI ; module load FSL ; source /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.3-centos7_64/etc/fslconf/fsl.sh ; module load dcm2niix ; . /gpfs/milgram/apps/hpc.rhel7/software/Python/Anaconda3/etc/profile.d/conda.sh ; conda activate harmonic

## get tunneling info
XDG_RUNTIME_DIR=""
ipnport=$(shuf -i8000-9999 -n1)
ipnip=$(hostname -i)
server=$(hostname)
## print tunneling instructions to jupyter-log-{jobid}.txt
echo -e "
    Copy/Paste this in your local terminal to ssh tunnel with remote
    -----------------------------------------------------------------
    ssh -N -L $ipnport:$ipnip:$ipnport $USER@milgram.hpc.yale.edu
    -----------------------------------------------------------------

    Then open a browser on your local machine to the following address
    ------------------------------------------------------------------
    localhost:$ipnport  (prefix w/ https:// if using password)
    ------------------------------------------------------------------
    "

## start an ipcluster instance and launch jupyter
jupyter notebook --no-browser --port=$ipnport --ip=$ipnip --notebook-dir /gpfs/milgram/project/turk-browne/ --NotebookApp.iopub_data_rate_limit=1.0e10
#jupyter-notebook --no-browser --port=$ipnport --ip=$ipnip --notebook-dir /gpfs/milgram/project/turk-browne/users/kp578/localize #/gpfs/milgram/project/turk-browne/projects/localize
