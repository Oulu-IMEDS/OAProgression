# Osteoarthritis Progression Prediction from Plain Radiographs: Data from OAI and MOST studies

*Codes for paper and the pre-trained models.*

(c) Aleksei Tiulpin, University of Oulu, 2018.

## About

This repository contains the full codes to reproduce the training process in the paper. To train the model from scratch, you need to obtain the DICOM images from MOST and OAI datasets. You also need to get the corresponding metadata.

## Installation, training and evaluation

### Dependencies

To run this project, we used `Ubuntu 16.04`, `Docker` and also `nvidia-docker`. These are the only software dependencies you will need. Please, install these and you are set.

For training we used 3xGTX1080Ti NVidia cards, so, please make sure that you have enough memory for minibatches allocation.

### Retraining

1. Download the SAS data from OAI website: https://ndar.nih.gov/oai/full_downloads.html (select `X-RAY ASSESSMENTS - SAS`).
2. Download the OAI subjects and clinical assessments data (select `GENERAL - SAS` and `ALL CLINICAL - SAS`), and place `enrollees.sas7bdat` and `allclinical00.sas7bdat`.
3. Place clinical and subject data into the folder downloaded in 1.
4. Copy MOST metadata into some folder and place `MOST_names.csv` there as well.
5. Set-up the metadata, localized ROI and workdir paths in `run_training_docker.sh`
6. Execute `run_training_docker.sh`

After sometime, the script will create a snapshot located in the specified workdir. Remember the name of this snapshot.

### Running the baselines evaluation

1. Specify the paths in `run_baselines_docker.sh` similarly as was done for `run_training_docker.sh`.
Do not forget to specify the snapshot paths for the model you have just trained and the name of the snapshot.
This is needed to use exectly the same cross-validation splits.
2. Execute `run_baselines_docker.sh`.

This script will generate multiple plots in teh workdir, which we used in the article.

### Running the neural networks

1. Similarly to the previous section, modify `run_eval_docker.sh` and specify all the paths.
2. Run `run_eval_docker.sh`

This script will generate multiple plots in teh workdir, which were also used in the paper.

## Testing on your own images

TBC

## License

This software and the pretrained models can be used only for research purposes


