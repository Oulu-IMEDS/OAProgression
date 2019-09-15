#!/bin/bash

OAI_META_SRC=/media/lext/FAST/OA_progression_project/Data/X-Ray_Image_Assessments_SAS
MOST_META_SRC=/media/lext/FAST/OA_progression_project/Data/most_meta
OAI_MOST_IMG_SRC=/media/lext/FAST/OA_progression_project/Data/MOST_OAI_00_0_2
WRKDIR=/media/lext/FAST/OA_progression_project/workdir

mkdir -p $WRKDIR/torch_models

docker build -t oaprog_img .

nvidia-docker run -it --name=oa_prog_data_preparation --rm \
	      -v $OAI_META_SRC:/oai_meta:ro \
	      -v $MOST_META_SRC:/most_meta:ro \
	      -v $OAI_MOST_IMG_SRC:/dataset:ro \
	      -v $WRKDIR:/workdir/:rw --ipc=host \
	      oaprog_img python -u prepare_metadata.py \
	      --oai_meta /oai_meta \
	      --most_meta /most_meta \
	      --imgs_dir /dataset \
	      --save_meta /workdir/Metadata

nvidia-docker run -it --name oa_prog_training --rm \
	      -v $WRKDIR:/workdir/:rw \
	      -v $OAI_MOST_IMG_SRC:/data/:ro \
	      -v $WRKDIR/torch_models:/root/.torch/models/ --ipc=host \
	      oaprog_img python -u run_training.py \
	      --snapshots /workdir/snapshots \
	      --logs /workdir/logs \
	      --dataset_root /data/ \
	      --metadata_root /workdir/Metadata \
	      --n_epochs 20