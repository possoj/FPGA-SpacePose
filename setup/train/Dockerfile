FROM nvcr.io/nvidia/pytorch:21.10-py3
# Pytorch 1.10.0 (https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_21-10.html#rel_21-10)

LABEL authors="Julien Posso"
LABEL description="Docker image for training mixed-precision quantized DNN with Pytorch and Brevitas"

# Avoid questions during build
ARG DEBIAN_FRONTEND="noninteractive"

ENV TZ="America/Montreal"

# Install python packages
# RUN conda install -y -c conda-forge xlsxwriter==3.0.3 onnx==1.11.0 onnxruntime==1.11.1 pandas==1.2.3 tensorboard==2.6.0
RUN pip install xlsxwriter==3.0.3 pandas==1.2.3 brevitas==0.7.1 yacs==0.1.8 torchlop==0.2 openpyxl==3.1.2
