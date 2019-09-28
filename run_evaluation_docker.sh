#!/bin/bash

WRKDIR=/media/lext/FAST/OA_progression_project/workdir
SNAPSHOT=2019_01_29_10_18 #$(ls -td $WRKDIR/snapshots/* | head -1 | rev |cut -d/ -f1 | rev)
OAI_MOST_IMG_SRC=/media/lext/FAST/OA_progression_project/Data/MOST_OAI_00_0_2

docker build -t oaprog_img .


echo "====> Working on the snapshot $SNAPSHOT"

## If you run it first time - remove the option "--from_cache".
#nvidia-docker run -it --name oa_progression_oof_inference --rm \
#	      -v $WRKDIR:/workdir/:rw \
#	      -v $OAI_MOST_IMG_SRC:/data/:ro --ipc=host \
#	      oaprog_img python -u run_oof_inference.py --snapshots /workdir/snapshots \
#	      --snapshot $SNAPSHOT \
#	      --dataset_root /data/ \
#	      --save_dir /workdir/Results \
#	      --metadata_root /workdir/Metadata \
# #        --from_cache True
#
## Running the "easy baseline" based on logreg
#nvidia-docker run -it --name oa_prog_logreg_baselines_eval --rm \
#	      -v $WRKDIR:/workdir/:rw --ipc=host \
#	      oaprog_img python -u run_logreg_baselines.py \
#	      --snapshots_root /workdir/snapshots \
#	      --snapshot $SNAPSHOT \
#	      --metadata_root /workdir/Metadata \
#	      --save_dir /workdir/Results
#
## Running the more complex baseline based on LightGBM and bayesian hyper-parameters tuning
#nvidia-docker run -it --name oa_prog_lgb_baselines_eval --rm \
#	      -v $WRKDIR:/workdir/:rw --ipc=host \
#	      oaprog_img python -u run_lgbm_baselines.py \
#	      --snapshots_root /workdir/snapshots \
#	      --snapshot $SNAPSHOT \
#	      --metadata_root /workdir/Metadata \
#	      --save_dir /workdir/Results
#
# If you run it first time - remove the option "--from_cache".
#nvidia-docker run -it --name oa_prog_evaluation --rm \
#	      -v $WRKDIR:/workdir/:rw \
#	      -v $OAI_MOST_IMG_SRC:/data/:ro --ipc=host \
#	      oaprog_img python -u run_dl_evaluation.py --snapshots /workdir/snapshots \
#	      --snapshot $SNAPSHOT \
#	      --dataset_root /data/ \
#	      --save_dir /workdir/Results \
#	      --metadata_root /workdir/Metadata \
#	      --from_cache True --plot_gcams True
#
## Running the stacked predictions
nvidia-docker run -it --name oa_prog_stacking_eval --rm \
	      -v $WRKDIR:/workdir/:rw --ipc=host \
	      oaprog_img python -u run_second_level_model.py \
	      --snapshots_root /workdir/snapshots \
	      --snapshot $SNAPSHOT \
	      --metadata_root /workdir/Metadata \
	      --save_dir /workdir/Results
#
## Generating the preliminary plots (just for viewing)
#nvidia-docker run -it --name oa_prog_eval_cmp --rm \
#	      -v $WRKDIR:/workdir/:rw --ipc=host \
#	      oaprog_img python -u run_models_comparison.py \
#	      --metadata_root /workdir/Metadata \
#	      --results_dir /workdir/Results
#
#echo "==> Evaluation results for KL0-1"
## Doing evaluation of the KL01 cases
#nvidia-docker run -it --name oa_prog_eval_kl01_cmp --rm \
#	      -v $WRKDIR:/workdir/:rw --ipc=host \
#	      oaprog_img python -u run_models_comparison_kl_01.py \
#	      --metadata_root /workdir/Metadata \
#	      --results_dir /workdir/Results
