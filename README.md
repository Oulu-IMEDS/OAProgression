# Multimodal Machine Learning-based Knee Osteoarthritis Progression Prediction from Plain Radiographs and Clinical Data

*Codes for paper and the pre-trained models.*

Arxiv pre-print: http://arxiv.org/abs/1904.06236

(c) Aleksei Tiulpin, University of Oulu, 2018-2019.

## About

<center>
<img src="https://github.com/mipt-oulu/oaprogression/blob/master/docs/schema.png" width="900"/> 
</center>

This repository contains the full codes to reproduce the training process in the paper. To train the model from scratch, you need to obtain the DICOM images from MOST and OAI datasets. You also need to get the corresponding metadata (downloadable from the website. More instructions on getting the data are provided in [documentation](docs/DATASETS.md). The metadata, required for annotation of these images is available at [our websie](http://mipt-ml.oulu.fi/datasets/OAProgression).

## Installation, training and evaluation

### Dependencies

To run this project, we used `Ubuntu 16.04`, `Docker` and also `nvidia-docker`. 
These are the only software dependencies you will need. Please, install these and you are set. 
If you do not want to use the `Docker`, you can follow teh given Dockerfile to follow the installation process.

We used 3xGTX1080Ti NVIDIA cards to train our models, so, please make sure that you have enough memory for minibatches allocation. We assume, that 2xGTX1080Ti will also be sufficient.

### Reproducing the experiments

1. Set-up the metadata, localized ROI and workdir paths in `run_training_docker.sh`. Do the same for `run_evaluation_docker.sh`
2. Execute `run_training_docker.sh`
3. Execute `run_evaluation_docker.sh`

Eventually, these scripts will generate a snapshot, containing 5 models from each cross-validation fold. 
You can monitor the process using tensorboard (needs to be run independently), or look at the training logs later, after the experiments. 
They will be generated in the `WRKDIR` (look at the `run_training_docker.sh`.

### Pre-trained models
The pre-trained CNN models are available at [http://mipt-ml.oulu.fi/models/OAProgression/](http://mipt-ml.oulu.fi/models/OAProgression/).
To train the second-level LightGBM models, you need to obtain the OAI dataset metadata.

## Citation
If you use any of this code or data, please cite the following paper:

```
@misc{1904.06236,
Author = {Aleksei Tiulpin and Stefan Klein and Sita M. A. Bierma-Zeinstra and Jérôme Thevenot and Esa Rahtu and Joyce van Meurs and Edwin H. G. Oei and Simo Saarakkala},
Title = {Multimodal Machine Learning-based Knee Osteoarthritis Progression Prediction from Plain Radiographs and Clinical Data},
Year = {2019},
Eprint = {arXiv:1904.06236},
}
```
## License
This software and the pre-trained models can be used only for research purposes.


