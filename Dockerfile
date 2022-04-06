#FROM nvidia/cuda:10.2-devel

# Miniconda install copy-pasted from Miniconda's own Dockerfile reachable
# at: https://github.com/ContinuumIO/docker-images/blob/master/miniconda3/debian/Dockerfile



#baseImage
FROM anibali/pytorch:1.8.1-cuda11.1-ubuntu20.04


#RUN apt-get update
#RUN apt-get upgrade -y
#RUN apt-get install -y python3-pip

#RUN apt-get install -y python
#RUN apt-get install -y python3-pip
#RUN apt-get -y install python3-pip

WORKDIR /cyclegan

COPY require.txt .
#install dependincies
RUN pip install -r require.txt

COPY . .

#FROM nvidia/cuda:10.2-base
#CMD nvidia-smi
#ENV NVIDIA_VISIBLE_DEVICES all
#ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

#specify the entrycomand
CMD ["python","train.py"]
#


