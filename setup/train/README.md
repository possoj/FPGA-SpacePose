# 1. Docker (recommended)

## 1.1. Build the docker image

command to build the docker container (from the project root folder):
```shell
docker build setup/train -t pose_brevitas:latest
```

## 1.2. Run the docker image

Command to run the docker image:
```shell
docker run --rm -it --gpus=all --ipc=host --net=host --entrypoint=bash --hostname=brevitas_pose -v $POSE_ESTIMATION_ROOT:/workspace/pose_estimation pose_brevitas:latest
```
Replace $POSE_ESTIMATION_ROOT with your pose_estimation path that contains both the pose_estimation project and 
the dataset.

With podman rootless:
```shell
podman run --rm -it --net=host --entrypoint bash --security-opt=label=disable --hooks-dir=/usr/share/containers/oci/hooks.d/-v /export/tmp/posso/recherche:/workspace/pose_estimation pose_brevitas:latest 
```

# 2. Anaconda

```shell
conda create --name pose_brevitas python=3.8.12
conda activate pose_brevitas
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install -c conda-forge tqdm==4.62.3 matplotlib==3.4.3 opencv==4.5.5
# conda install -c conda-forge xlsxwriter==3.0.3 onnx==1.11.0 onnxruntime==1.11.1 pandas==1.2.3 tensorboard==2.6.0
pip install xlsxwriter==3.0.3 onnx==1.11.0 onnxruntime==1.11.1 pandas==1.2.3 tensorboard==2.6.0 protobuf==3.18.1
pip install brevitas==0.7.1 onnxoptimizer==0.2.7 yacs==0.1.8
```


