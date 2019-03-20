#!/bin/bash

WRKDIR=/media/lext/FAST/OA_progression_project/workdir
SNAPSHOT=$(ls -td $WRKDIR/snapshots/* | head -1 | rev |cut -d/ -f1 | rev)
OAI_MOST_IMG_SRC=/media/lext/FAST/OA_progression_project/Data/MOST_OAI_00_0_2

docker build -t oaprog_img .

# Generating the plots
nvidia-docker run -it --name oa_prog_eval_cmp --rm \
	      -v $WRKDIR:/workdir/:rw --ipc=host \
	      oaprog_img python -u generate_paper_subplots.py \
	      --metadata_root /workdir/Metadata \
	      --results_dir /workdir/Results