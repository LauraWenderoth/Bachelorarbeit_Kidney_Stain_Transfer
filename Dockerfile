FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

COPY requirments.txt /

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,video,utility

RUN apt-get update \
#  && apt-get install --no-install-recommends ffmpeg libsm6 libxext6 git gcc openslide-tools python-openslide -y \
  && pip install -U pip setuptools wheel \
  && pip install --no-cache-dir -r /requirments.txt
