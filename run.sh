#!/bin/bash

#SBATCH -O $root$/$blockname$/slurm_log/$simid$.%j.%N
#SBATCH -D $root$/fargo3d/
#SBATCH -J $simid$.%j.%N
#SBATCH -p light
#SBATCH --get-user-env
#SBATCH --mail-type=end
#SBATCH --mail-user=alessandro.ruzza@unimi.it
#SBATCH --mem=$ram$
#SBATCH --account=evaluation
#SBATCH -c $cores$
#SBATCH --time=72:00:00

spack load intel-oneapi-mpi@2021.9.0
export FI_LOG_LEVEL=debug
export I_MPI_PMI_LIBRARY="/lib64/libpmi.so"
export UCX_TLS=all
#Sexport UCX_TLS=tcp,ud,sm,self


make clean SETUP=$setup$
make para SETUP=$setup$

srun ./fargo3d $root$/$blockname$/params/para_$simid$.par







