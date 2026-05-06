#!/bin/bash

# Time estimate:
#   ~20 min per 1000 episodes  →  30,000 episodes ≈ 600 min ≈ 10 h
#   Wall time set to 24 h to leave a safety buffer.
#
# Resume behaviour:
#   If the job times out or crashes, just resubmit this same script.
#   Training will automatically resume from the latest checkpoint in
#   checkpoints/hrep/.

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=amilan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --job-name=inter-hrep
#SBATCH --output=inter-hrep.%j.out
#SBATCH --qos=normal

module purge
module load anaconda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amfa-custom-env

python3 run_hrep.py \
    --episodes 30000 \
    --checkpoint-dir checkpoints/hrep \
    --checkpoint-interval 500
