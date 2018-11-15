#!/bin/bash

WRKDIR=/media/lext/FAST/OA_progression_project/workdir
SNAPSHOT=2018_11_15_14_39

docker build -t oaprog_img .

nvidia-docker run -it --name oa_prog_baselines_eval --rm \
	      -v $WRKDIR:/workdir/:rw --ipc=host \
	      oaprog_img python -u run_baselines.py --snapshots_root /workdir/snapshots --snapshot $SNAPSHOT --metadata_root /workdir/Metadata --save_dir /workdir/Results         
