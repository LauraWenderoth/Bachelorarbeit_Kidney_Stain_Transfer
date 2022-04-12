##FROM nvidia/cuda:10.2-devel
#
## Miniconda install copy-pasted from Miniconda's own Dockerfile reachable
## at: https://github.com/ContinuumIO/docker-images/blob/master/miniconda3/debian/Dockerfile
#
#
#
##baseImage
#FROM nvidia/cuda:11.4.0-base-ubuntu18.04
###11.5.1-cudnn8-runtime-ubuntu18.04
##
###RUN apt-get update && apt-get install -y python3 python3-pip sudo
###anibali/pytorch:1.8.1-cuda11.1-ubuntu20.04
##
##
###RUN apt-get update
##RUN apt-get upgrade
###RUN pip3 install --upgrade pip==20.0.1
###RUN apt-get install -y \
###    sudo \
###    python3.8 \
###    python3-pip
##
##RUN apt-get update && \
##    apt-get install --no-install-recommends -y \
##    python3.8 python3-pip python3.8-dev
###RUN apt-get install -y python3-pip
##
###RUN apt-get install -y python
###RUN apt-get install -y python3-pip
###RUN apt-get -y install python3-pip
### RUN venv /env -p python3.7
##WORKDIR /cyclegan
##
##COPY require.txt .
###install dependincies
##RUN pip install -r require.txt
##
##COPY . .
##
###FROM nvidia/cuda:10.2-base
###CMD nvidia-smi
###ENV NVIDIA_VISIBLE_DEVICES all
###ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
##
###specify the entrycomand
##CMD ["python","train.py"]
#
#
#RUN apt-get update && \
#        apt-get install -y software-properties-common vim
#        add-apt-repository ppa:jonathonf/python-3.6
#RUN apt-get update -y
#
#RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv && \
#        apt-get install -y git
#
## update pip
#RUN python3.6 -m pip install pip --upgrade && \
#        python3.6 -m pip install wheel
#
#WORKDIR /cyclegan
#COPY require.txt .
#RUN pip install -r require.txt
#COPY . .


FROM pytorch/pytorch:1.11.0-cuda11.3-cudexitnn8-runtime

COPY ./require.txt /

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,video,utility

RUN apt-get update \
#  && apt-get install --no-install-recommends ffmpeg libsm6 libxext6 git gcc openslide-tools python-openslide -y \
  && pip install -U pip setuptools wheel \
  && pip install --no-cache-dir -r /require.txt
