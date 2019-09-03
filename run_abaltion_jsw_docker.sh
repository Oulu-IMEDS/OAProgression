#!/bin/bash

DATA=/media/lext/FAST/OA_progression_project/Data/
WRKDIR=/media/lext/FAST/OA_progression_project/workdir

mkdir -p $WRKDIR

docker build -t oaprog_img .

nvidia-docker run -it --name oa_prog_jsw_ablation --rm \
	      -v $WRKDIR:/workdir/:rw \
	      -v $DATA:/data/:ro --ipc=host \
	      oaprog_img python -u run_jsw_ablation_experiments.py \
	      --dataset_root /data/ \
	      --metadata_root /workdir/Metadata