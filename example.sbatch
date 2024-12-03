#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --constraint=rtx_4090               ### Available values:  cpu, gpu, cpu128, cpu256, gtx_1080, rtx_2080, rtx_3090, rtx_4090, rtx_6000, titan_rtx, tesla_p100 
#SBATCH --partition main		        	### specify partition name where to run a job. change only if you have a matching qos!! main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time 7-0:0:00			            ### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name kitti2sceneflow_clip_16000_1e3_fresh_MLP.out   		### name of the job
#SBATCH --output %J_kitti2sceneflow_clip_16000_1e3_fresh_MLP.out	### output log for running job - %J for job number
#SBATCH --gpus=rtx_4090:1			    	### number of GPUs, allocating more than 1 requires IT team's permission. Example to request 3090 gpu: #SBATCH --gpus=rtx_3090:1

# Note: the following 4 lines are commented out
##SBATCH --mail-user=alonkay@post.bgu.ac.il	### user's email for sending job status messages
##SBATCH --mail-type=ALL			### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
##SBATCH --mem=12				### ammount of RAM memory, allocating more than 60G requires IT team's permission

################  Following lines will be executed by a compute node    #######################

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

### Start your code below ####
module load anaconda				### load anaconda module (must be present when working with conda environments)
source activate alon_env				### activate a conda environment, replace my_env with your conda environment
python Main.py			### run my_code.py python file and send my_arg1 argument to it


####################################### FOR ME #########################################################
## squeue --me
## sbatch example.sbatch
## scontrol show job _
## sacct -j _ --format=JobName,MaxRSS,AllocTRES,State,Elapsed,Start,ExitCode,DerivedExitcode,Comment   # Completed job information 
####################################### FOR ME #########################################################
