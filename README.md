# Osteoarthritis Progression Prediction from Plain Radiographs

Codes for paper and the pre-trained models.

(c) Aleksei Tiulpin, University of Oulu, Erasmus MC, 2018.

## About

This repository contains the full codes to reproduce the training process in the paper. To train the model from scratch, you need to obtain the DICOM images from MOST and OAI datasets. You also need to get the corresponding metadata.

## Installation

TODO

### Conda environment

TODO

### Metadata preparation

1. Download the SAS data from OAI website: https://ndar.nih.gov/oai/full_downloads.html (select `X-RAY ASSESSMENTS - SAS`).
2. Extract the ZIP archive and store in some location, which we denote as `OAI_SRC_DIR/`.
This path should include `X-Ray Image Assessments_SAS/`.
3. Download the OAI subjects and clinical assessments data 
(select `GENERAL - SAS` and `ALL CLINICAL - SAS`), and place `enrollees.sas7bdat` and `allclinical00.sas7bdat` to `OAI_SRC_DIR`.
4. Place the metadata from MOST dataset to `MOST_SRC_DIR`
5. Copy the file `MOST_names.csv` to `MOST_SRC_DIR`.
5. Run the script `prepare_metadata.py --oai_meta OAI_SRC_DIR --most_meta MOST_SRC_DIR`

The script will create the files `OAI_progression.csv`, `OAI_clinical.csv`, `MOST_progression.csv` 
and `MOST_clinical.csv`in the folder `Metadata`.

### Running the models

TODO


