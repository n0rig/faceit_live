FROM tensorflow/tensorflow:latest-gpu-py3
ENV LANG C.UTF-8


###########
##  faceswap dockerfile ##
###########

RUN apt-get update -qq -y \
 && apt-get install -y libsm6 libxrender1 libxext-dev python3-tk libcups2-dev \
 && apt-get install -y libxml2-dev libxslt-dev python-dev python3-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/
RUN pip3 install cmake
RUN pip3 --no-cache-dir install -r /opt/requirements.txt && rm /opt/requirements.txt


# patch for tensorflow:latest-gpu-py3 image
RUN cd /usr/local/cuda/lib64 \
 && mv stubs/libcuda.so ./ \
 && ln -s libcuda.so libcuda.so.1 \
 && ldconfig

###########
## Tools ##
###########

RUN apt-get update --fix-missing && apt-get install -y \
    wget \
    vim \
    git \
    unzip \
    cmake \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

##############
## Anaconda ##
##############

RUN apt-get update --fix-missing && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda3-2019.03-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

ENV PATH /opt/conda/bin:$PATH

#########################
## faceswap_live dependencies ##
#########################

#RUN git clone --recurse-submodules https://github.com/alew3/faceit_live.git /code/faceit_live


WORKDIR /code/faceit_live

# Solves: `libjpeg.so.8: cannot open shared object file: No such file or directory`
#          after `from PIL import Image`
RUN apt-get install -y libjpeg-turbo8 \
    && rm -rf /var/lib/apt/lists/*

RUN echo export CUDA_DEVICE_ORDER="PCI_BUS_ID" >> ~/.bashrc

# https://software.intel.com/en-us/mkl
