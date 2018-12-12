# Osteoarthritis Progression Prediction from Plain Radiographs: Data from OAI and MOST studies

*Codes for paper and the pre-trained models.*

(c) Aleksei Tiulpin, University of Oulu, 2018.

<img src="https://github.com/mipt-oulu/oaprogression/blob/master/docs/schema.png" width="600"/> 

## About

This repository contains the full codes to reproduce the training process in the paper. To train the model from scratch, you need to obtain the DICOM images from MOST and OAI datasets. You also need to get the corresponding metadata (downloadable from the website. More instructions on getting the data are provided in [documentation](docs/DATASETS.md). The metadata, required for annotation of these images is available at [our websie](http://mipt-ml.oulu.fi/datasets/OAProgression).

## Installation, training and evaluation

### Dependencies

To run this project, we used `Ubuntu 16.04`, `Docker` and also `nvidia-docker`. These are the only software dependencies you will need. Please, install these and you are set.

We used 3xGTX1080Ti NVIDIA cards to train our models, so, please make sure that you have enough memory for minibatches allocation. We assume, that 2xGTX1080Ti will also be sufficient.

### Reproducing the experiemnts

1. Set-up the metadata, localized ROI and workdir paths in `run_training_docker.sh`. Do the same for `run_evaluation_docker.sh`
2. Execute `run_training_docker.sh`
3. Execute `run_evaluation_docker.sj`

Eventually, these scripts will generate a snapshot, containing 5 models from each cross-validation fold. You can monitor the process using tensorboard (needs to be run independently), or look at the training logs later, after the experiments. They will be generated in the `WRKDIR` (look at the `run_training_docker.sh`.

## License

This software and the pretrained models can be used only for research purposes


