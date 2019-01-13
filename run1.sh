#!/bin/bash	
#SBATCH	-J	more_data
#SBATCH	-p	gpu	
#SBATCH	-N	1	
#SBATCH	--output=log.%j.out	
#SBATCH	--error=log.%j.err	
#SBATCH	-t	60:00:00	
#SBATCH	--gres=gpu:1
module	load	anaconda3/5.3.0	
python generate_fingerprint.py
