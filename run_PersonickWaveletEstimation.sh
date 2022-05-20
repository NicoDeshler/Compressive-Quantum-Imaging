#!/bin/bash
#PBS -N QMWaveletEst_8x8
#PBS -j oe
#PBS -d /home/nicolas/Research/Compressive-Quantum-Imaging/
#PBS -l nodes=1:ppn=8
#PBS -l walltime=120:00:00
#PBS -l mem=16GB
matlab "PersonickWaveletEstimation(1,'ArrayID', getenv('PBS_ARRAYID'), 'JobID', getenv('PBS_JOBID'));"
