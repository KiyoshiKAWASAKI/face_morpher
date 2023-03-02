#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q gpu -l gpu=1
#$ -l h=!qa-a10-005&!qa-rtx6k-044&!qa-a10-006
#$ -e errors/
#$ -N

# Required modules
module load conda
conda init bash
source activate face_morph2

export DLIB_DATA_DIR=/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_data/shape_predictor_68_face_landmarks.dat
python facemorpher/morpher.py