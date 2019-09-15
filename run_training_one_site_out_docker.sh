#!/bin/bash

OAI_SRC=/media/lext/FAST/OA_progression_project/Data/
OAI_MOST_IMG_SRC=/media/lext/FAST/OA_progression_project/Data/MOST_OAI_00_0_2
WRKDIR=/media/lext/FAST/OA_progression_project/workdir

mkdir -p $WRKDIR/torch_models

docker build -t oaprog_img .

nvidia-docker run -it --name oa_prog_training --rm \
	      -v $WRKDIR:/workdir/:rw \
	      -v $OAI_MOST_IMG_SRC:/data/:ro \
	      -v $OAI_SRC:/oai_data_root:ro \
	      -v $WRKDIR/torch_models:/root/.torch/models/ --ipc=host \
	      oaprog_img python -u run_crossval_across_sites.py \
	      --snapshots /workdir/snapshots \
	      --logs /workdir/logs \
	      --dataset_root /data/ \
	      --metadata_root /workdir/Metadata \
	      --oai_data_root /oai_data_root \
	      --n_epochs 20