#!/bin/bash

WRKDIR=/media/lext/FAST/OA_progression_project/workdir
SNAPSHOT=2018_11_15_14_39
OAI_MOST_IMG_SRC=/media/lext/FAST/OA_progression_project/Data/MOST_OAI_00_0_2

docker build -t oaprog_img .

nvidia-docker run -it --name oa_prog_baselines_eval --rm \
	      -v $WRKDIR:/workdir/:rw --ipc=host \
	      oaprog_img python -u run_baselines.py --snapshots_root /workdir/snapshots --snapshot $SNAPSHOT --metadata_root /workdir/Metadata --save_dir /workdir/Results

# If you run it first time - remove the option "--from_cache".
nvidia-docker run -it --name oa_prog_evaluation --rm \
	      -v $WRKDIR:/workdir/:rw \
	      -v $OAI_MOST_IMG_SRC:/data/:ro --ipc=host \
	      oaprog_img python -u run_evaluation.py --snapshots /workdir/snapshots --snapshot $SNAPSHOT --dataset_root /data/ --save_dir /workdir/Results --metadata_root /workdir/Metadata --from_cache True

