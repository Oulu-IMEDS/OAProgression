FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

MAINTAINER Aleksei Tiulpin, University of Oulu, Version 3.0

RUN apt-get update
RUN apt-get install -y libgtk2.0-dev
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
	     unzip \
	     zip \
	     locales \
	     emacs \
	     libgl1-mesa-glx \
	     openssh-server \
	     screen \
	     libturbojpeg \
	     rsync \
         wget


RUN locale-gen --purge en_US.UTF-8
RUN echo -e 'LANG="en_US.UTF-8"\nLANGUAGE="en_US:en"\n' > /etc/default/locale
RUN dpkg-reconfigure --frontend=noninteractive locales

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN chmod +x ~/miniconda.sh && ~/miniconda.sh -b -p /opt/conda && rm ~/miniconda.sh
ENV PATH=/opt/conda/bin:${PATH}
RUN conda update -n base conda

RUN conda create -y --name oaprog python=3.6
RUN conda install -y pytorch=0.4.1 torchvision cuda91 -c pytorch -n oaprog

COPY requirements.txt requirements.txt
ENV PATH /opt/conda/envs/oaprog/bin:$PATH
RUN pip install pip -U -v && pip install -r requirements.txt

# Installing the other deps
RUN conda install -y numpy=1.15.2 scipy=1.0.1 matplotlib -n oaprog
RUN conda install -y opencv=3.4.1 -c conda-forge -n oaprog

# Fixing the matplotlib backend issues
RUN mkdir -p /root/.config/matplotlib/
RUN echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

# Setting up the package
RUN mkdir /opt/pkg
COPY . /opt/pkg
RUN pip install -e /opt/pkg/

# Copying the files
RUN cp /opt/pkg/scripts/prepare_metadata.py .
RUN cp /opt/pkg/scripts/run_training.py .
RUN cp /opt/pkg/scripts/run_logreg_baselines.py .
RUN cp /opt/pkg/scripts/run_lgbm_baselines.py .
RUN cp /opt/pkg/scripts/run_evaluation.py .
RUN cp /opt/pkg/scripts/run_oof_inference.py .
RUN cp /opt/pkg/scripts/run_second_level_model.py .