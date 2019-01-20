#!/bin/bash

OAI_META_SRC=/media/lext/FAST/OA_progression_project/Data/X-Ray_Image_Assessments_SAS
MOST_META_SRC=/media/lext/FAST/OA_progression_project/Data/most_meta
OAI_MOST_IMG_SRC=/media/lext/FAST/OA_progression_project/Data/MOST_OAI_00_0_2
WRKDIR=/media/lext/FAST/OA_progression_project/workdir

mkdir -p $WRKDIR

docker build -t oaprog_img .

nvidia-docker run -it --name oa_prog_training --rm \
	      -v $WRKDIR:/workdir/:rw \
	      -v $OAI_MOST_IMG_SRC:/data/:ro --ipc=host \
	      oaprog_img python -u run_training_age_sex_bmi.py \
	      --snapshots /workdir/snapshots_age_sex_bmi \
	      --logs /workdir/logs_age_sex_bmi \
	      --dataset_root /data/ \
	      --metadata_root /workdir/Metadata \
	      --predict_age_sex_bmi True \
          --lr 1e-3 \
          --n_epochs 30