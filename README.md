# FPGA-SpacePose: Real-Time Spacecraft Pose Estimation

This repository contains the implementation of the paper titled **"Real-Time Spacecraft Pose Estimation Using 
Mixed-Precision Quantized Neural Network on COTS Reconfigurable MPSoC"** by Julien Posso, Guy Bois, and Yvon Savaria. 
The project leverages a mixed-precision quantized neural network to achieve real-time pose estimation of spacecraft using 
FPGA components of a Xilinx MPSoC. Our methodology includes a novel evaluation technique for layer-wise sensitivity to 
quantization, optimizing accuracy, latency, and FPGA resource utilization. The resulting implementation is significantly 
faster and more energy-efficient compared to existing solutions, marking a substantial advancement in the field.

For more details, please refer to the [ArXiv pre-print](https://arxiv.org/abs/2407.06170) / [IEEE Xplore paper](https://ieeexplore.ieee.org/document/10666317). 
This work was presented at IEEE NEWCAS 2024, a peer-reviewed conference.

```
@INPROCEEDINGS{10666317,
  author={Posso, Julien and Bois, Guy and Savaria, Yvon},
  booktitle={2024 22nd IEEE Interregional NEWCAS Conference (NEWCAS)}, 
  title={Real-Time Spacecraft Pose Estimation Using Mixed-Precision Quantized Neural Network on COTS Reconfigurable MPSoC}, 
  year={2024},
  volume={},
  number={},
  pages={358-362},
  keywords={Space vehicles;Quantization (signal);Source coding;Pose estimation;Neural networks;Real-time systems;Energy efficiency;Artificial Intelligence;Neural Networks;Mixed-Precision Quantization;Inference;Embedded Systems;Aerospace},
  doi={10.1109/NewCAS58973.2024.10666317}}
```

We have also provided the poster presented at the conference for a visual summary of our work. You can view it below:
![Poster IEEE NEWCAS 2024 Conference](poster/ieee_newcas_2024_poster.png "Poster")

## Installation

### Data and Directory Structure

We use the [SPEED dataset](https://kelvins.esa.int/satellite-pose-estimation-challenge/data/) released by Stanford SLAB and ESA. 
Just download the dataset and follow the directory structure (or modify the DATASET_PATH variable in the configuration files):

```
    ├── datasets            # Datasets folder
    │   ├── speed           # SPEED dataset
    │   │   ├── images      # SPEED images folder
    │   │   └──...          # JSON files, license, etc...
    │   └── others          # Potentially other Spacecraft Pose Estimation datasets
    │
    ├── FPGA-SpacePose      # Project root directory (Git)
    │   ├── data            # Contains the train/valid dataset split (JSON files)
    │   ├── experiments     # Folders that store the results of the training and build
    │   ├── finn_build      # Folder that contains the intermediate files of the build of the accelerator with FINN. Useful for debugging. Advise: empty the directory before each build
    │   ├── models          # Weights of our models used in the article
    │   ├── poster          # Folder that contains the poster presented at IEEE NEWCAS 2024
    │   ├── setup           # README and Dockerfiles to set up the environment to train the neural network and build the accelerator     
    │   ├── src             # Python source files
    │   └── ...             # License and README
```

To reproduce our results, you need the same train/valid split as us. 
Copy the two JSON files in the `data/speed` folder into the `datasets/speed/` folder along with the original JSON files.

### Build the Environment

See the README in the `setup/train` and `setup/build` folders.

## Train the Neural Network

Configure the YAML file in `src/config/train/exp_0/` and the corresponding bit-width JSON file if Brevitas is used. 
If you create multiple `exp_...` directories, it will run multiple trainings. 
The current YAML file is set to reproduce the experiment of the published article. 
The training script creates corresponding folders in `experiments/train` to store the results and configuration files for reproducibility.

To train the neural network, launch the [train docker image](setup/train/README.md), go to the project root folder, and launch the training script.
```shell
python train.py
```

This table explains the various settings in the YAML configuration file used for training a neural network.

| Section | Parameter                 | Description                                                                      | Example Value                                    |
|---------|---------------------------|----------------------------------------------------------------------------------|--------------------------------------------------|
| `DATA`  | `BATCH_SIZE`              | Number of samples processed before the model is updated.                         | `32`                                             |
| `DATA`  | `IMG_SIZE`                | Size of the input images in pixels (height, width).                              | `[240, 240]`                                     |
| `DATA`  | `ORI_SMOOTH_FACTOR`       | Smoothing factor for orientation soft-classification.                            | `3`                                              |
| `DATA`  | `ROT_AUGMENT`             | Flag to apply manual rotation augmentation with OpenCV.                          | `true`                                           |
| `DATA`  | `OTHER_AUGMENT`           | Flag to apply other data augmentations.                                          | `true`                                           |
| `DATA`  | `PATH`                    | Path to the dataset directory.                                                   | `../datasets/speed`                              |
| `DATA`  | `SHUFFLE`                 | Flag to shuffle the dataset before each epoch.                                   | `true`                                           |
| `MODEL` | `BACKBONE.NAME`           | Name of the model backbone architecture.                                         | `mobilenet_v2_brevitas`, `mobilenet_v2_pytorch`  |
| `MODEL` | `BACKBONE.RESIDUAL`       | Flag to use residual connections in the backbone.                                | `true`                                           |
| `MODEL` | `HEAD.NAME`               | Name of the head architecture.                                                   | `ursonet_pytorch`, `ursonet_brevitas`            |
| `MODEL` | `HEAD.N_ORI_BINS_PER_DIM` | Number of orientation bins per dimension in the classification head.             | `12`                                             |
| `MODEL` | `HEAD.ORI`                | Type of orientation prediction (classification or regression).                   | `classification`, `regression`                   |
| `MODEL` | `HEAD.POS`                | Type of position prediction (only regression is implemented).                    | `regression`                                     |
| `MODEL` | `MANUAL_COPY`             | Flag to manually copy model parameters (from PyTorch to Brevitas model).         | `true`                                           |
| `MODEL` | `PRETRAINED_PATH`         | Path to the pretrained model parameters.                                         | `models/mobile_ursonet_fp32/model/parameters.pt` |
| `MODEL` | `QUANTIZATION`            | Flag to apply quantization to the model (only if Brevitas backbone and/or head). | `true`                                           |
| `TRAIN` | `N_EPOCH`                 | Number of epochs to train the model.                                             | `20`                                             |
| `TRAIN` | `LR`                      | Initial learning rate.                                                           | `0.01`                                           |
| `TRAIN` | `DECAY`                   | Weight decay (L2 penalty) to apply to the optimizer.                             | `0`                                              |
| `TRAIN` | `OPTIM`                   | Optimization algorithm to use.                                                   | `SGD`, `Adam`                                    |
| `TRAIN` | `MOMENTUM`                | Momentum factor for the optimizer.                                               | `0.9`                                            |
| `TRAIN` | `SCHEDULER`               | Learning rate scheduler to adjust the learning rate during training.             | `MultiStepLR`, `OnPlateau`                       |
| `TRAIN` | `GAMMA`                   | Multiplicative factor of learning rate decay.                                    | `0.1`                                            |
| `TRAIN` | `MILESTONES`              | Epochs at which to adjust the learning rate.                                     | `[7, 15]`                                        |

## Build the Neural Network FPGA Accelerator

### Configuration 

Configure the YAML file in `src/config/build/`. If you create multiple `exp_[X].yaml` files, it will run multiple builds. 
The current YAML file is set to reproduce the experiment of the published article. 
The accelerator use a trained neural network. We copy-pasted the folder generated by the training script from `experiments/train` to the `model/` folder. 

This table explains the various settings in the YAML configuration file used for configuring the build of a custom FPGA accelerator with FINN.

| Section | Parameter                       | Description                                                                                        | Example Value                             |
|---------|---------------------------------|----------------------------------------------------------------------------------------------------|-------------------------------------------|
| `DATA`  | `BATCH_SIZE`                    | Number of samples processed per batch.                                                             | `1`                                       |
| `DATA`  | `IMG_SIZE`                      | Size of the input images in pixels (height, width). Only squared images supported.                 | `[240, 240]`                              |
| `DATA`  | `ORI_SMOOTH_FACTOR`             | Smoothing factor for orientation soft-classification.                                              | `3`                                       |
| `DATA`  | `PATH`                          | Path to the dataset directory.                                                                     | `../datasets/speed`                       |
| `FINN`  | `ACCEL.CLK_PERIOD_NS`           | Clock period in nanoseconds for the accelerator.                                                   | `5`                                       |
| `FINN`  | `ACCEL.TARGET_CYCLES_PER_FRAME` | Target number of cycles per frame for the accelerator.                                             | `800000`                                  |
| `FINN`  | `BOARD.DEPLOYMENT_FOLDER`       | Folder path for deployment on the FPGA board.                                                      | `/home/xilinx/spacecraft_pose_estimation` |
| `FINN`  | `BOARD.NAME`                    | Name of the target FPGA board.                                                                     | `ZCU104`, `Ultra96`                       |
| `FINN`  | `FIFO.AUTO_DEPTH`               | Flag to automatically determine the depth of FIFOs (mandatory if shortcuts in the neural network). | `true`                                    |
| `FINN`  | `FIFO.SIZING_METHOD`            | Method for sizing the FIFOs (simulation or characterization).                                      | `largefifo_rtlsim`, `characterize`        |
| `FINN`  | `FIFO.RTL_SIM`                  | If true, use Python simulation instead of C++ RTL SIM.                                             | `true`                                    |
| `FINN`  | `FIFO.SPLIT_LARGE`              | Flag to allow very large FIFOs (large FIFOs are split into multiple small ones).                   | `true`                                    |
| `MODEL` | `BACKBONE.NAME`                 | Name of the model backbone architecture (must be a Brevitas backbone).                             | `mobilenet_v2_brevitas`                   |
| `MODEL` | `BACKBONE.RESIDUAL`             | Flag to use residual connections in the backbone.                                                  | `true`                                    |
| `MODEL` | `HEAD.NAME`                     | Name of the head architecture.                                                                     | `ursonet_pytorch`                         |
| `MODEL` | `HEAD.N_ORI_BINS_PER_DIM`       | Number of orientation bins per dimension in classification.                                        | `12`                                      |
| `MODEL` | `HEAD.ORI`                      | Type of orientation prediction (classification or regression).                                     | `classification`, `regression`            |
| `MODEL` | `HEAD.POS`                      | Type of position prediction (only regression is implemented).                                      | `regression`                              |
| `MODEL` | `MANUAL_COPY`                   | Flag to manually copy model parameters during training (should be false).                          | `false`                                   |
| `MODEL` | `PATH`                          | Path to the model directory that contains parameters and bit-width.                                | `models/mobile_ursonet_mpq/model`         |
| `MODEL` | `QUANTIZATION`                  | Flag to apply quantization to the model (must be true, FP32 models are not supported by FINN).     | `true`                                    |

For more details about the FINN Dataflow configuration, see the `build.py` script and the following file 
`/tools/finn/src/finn/builder/build_dataflow_config.py` in the [build Docker image](setup/build/README.md).

### Build Custom FPGA Accelerator

To build the neural network FPGA accelerator, launch the [build docker image](setup/build/README.md), 
go to the project root folder, and launch the build script.
```shell
python build.py
```

The build script uses our own folding algorithm. The FINN algorithm (version 0.8 was used during project development) 
produces folding configuration that violates the data-width converter rule which creates an error in later build steps, 
preventing an automatic end-to-end build. Our algorithm solves this issue (see `src/finn/folding.py`). 
It is compatible with the MobileNetV2 architecture. Other architectures may need to modify this script.

The build script uses our own transformation sequence (see `src/finn/build_steps.py`) with true residual connections, 
contrary to the FINN ResNet50 example which introduces convolutions and/or activation functions in the residual branch. 
The transformation sequence is compatible with the MobileNetV2 architecture. Other architectures may need to modify this script.

### Files Produced by the Build Script

The build script produces files in two folders. The `finn_build` folder contains every intermediate file that is necessary to build
the neural network accelerator. We recommend to delete the content of this folder before each build. The build script 
also generates the final results under the `experiments/build` folder. It contains the YAML configuration file, the 
parameters and bit-width of the source neural network (`model` folder) and the ONNX graph of the neural network (`export folder`). 
The files produced by the FINN compilation are generated in the `finn_output` folder. It contains:

[//]: # (- TODO: other folders)
- `build_dataflow.log`: the log of the build.
- `intermediate_models`: a folder that contains the ONNX intermediate models after each step.
- `report`: a folder that contains the reports generated by the `step_generate_estimate_reports` step:
  - `op_and_param_counts.json`: per layer number of parameters and MAC operations.
  - `estimate_layer_cycles.json`: per layer latency in clock cycles.
  - `estimate_layer_resources.json`: per layer FPGA Python resources estimation. 
  - `estimate_layer_resources_hls.json`: per layer FPGA resources report after synthesis (in finn_build/code_gen_ipgen...). 
  - `estimate_layer_config_alternatives.json`: list per layer FPGA resources estimation with all possible allocation: DSP, LUT, etc...
  - `estimate_network_perforamce.json`: 
    - max_cycles: latency of the slowest node in clock cycles
    - max_cycles_node_name : name of slowest node
    - estimated_throughput_fps = (10**9/clk_period_ns) / max_cycles
    - critical_path_cycles: pessimistic expected latency from input to output in clock cycles (sum of the latency of each node in the slowest branch)
    - estimated_latency_ns: input to output latency in clock cycles

## Deploy

To deploy the neural network FPGA accelerator on the Xilinx board, follow the section two in `setup/build/README.md`,
power-on the Xilinx board, launch the [build docker image](setup/build/README.md), go to the project root folder, and launch the deploy script.
```shell
python deloy.py
```

Enter the name of the folder that contains the neural network accelerator in `experiments/build` folder (e.g. `mobile_ursonet`). 
It will automatically send the files to the board, execute the scripts that program the FPGA with the bitfile, and bring back the results the computer. 


## Run the Bit-Width Experiment

Launch the [train docker image](setup/train/README.md) and go to the project root folder.

Generate the configuration files for the experiment.
```shell
python src/config/train/bit_width_experiment/generate_experiment.py
```

Ensure the `src/config/train/` folder does not contain any folder starting with `exp_` or YAML file starting with `exp_`. 
Then, copy-paste the content of the generated `src/config/train/bit_width_experiment/configs` folder into the `src/config/train/` folder.

Ensure the `experiments/train` folder does not contain any folder starting with `exp_`. Then run the experiment.
```shell
python train.py
```

Copy the results of the experiment (all folders starting with `exp_` in the `experiments/train` folder) into 
`experiments/train/bit_width_experiment/raw_result`.

Generate the figures.
```shell
python experiments/train/bit_width_experiment/bit_width_experiment.py
```
