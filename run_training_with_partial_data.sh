#!/bin/bash

OAI_SRC=/media/lext/FAST/OA_progression_project/Data/
OAI_MOST_IMG_SRC=/media/lext/FAST/OA_progression_project/Data/MOST_OAI_00_0_2
WRKDIR=/media/lext/FAST/OA_progression_project/workdir

mkdir -p $WRKDIR/torch_models

docker build -t oaprog_img .

for N_SAMPLES in 400 800 1600 3200;
do
    nvidia-docker run -it --name oa_prog_training --rm \
              -v $WRKDIR:/workdir/:rw \
              -v $OAI_MOST_IMG_SRC:/data/:ro \
              -v $WRKDIR/torch_models:/root/.torch/models/ --ipc=host \
              oaprog_img python -u run_training.py \
              --snapshots /workdir/snapshots \
              --subsample_train $N_SAMPLES \
              --logs /workdir/logs \
              --dataset_root /data/ \
              --metadata_root /workdir/Metadata \
              --n_epochs 20

    SNAPSHOT=$(ls -td $WRKDIR/snapshots/* | head -1 | rev |cut -d/ -f1 | rev)

    nvidia-docker run -it --name oa_prog_evaluation --rm \
	      -v $WRKDIR:/workdir/:rw \
	      -v $OAI_MOST_IMG_SRC:/data/:ro --ipc=host \
	      oaprog_img python -u run_dl_evaluation.py --snapshots /workdir/snapshots \
	      --snapshot $SNAPSHOT \
	      --dataset_root /data/ \
	      --save_dir /workdir/snapshots/$SNAPSHOT/test_inference \
	      --metadata_root /workdir/Metadata
done;