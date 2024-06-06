# 1. Installing FINN on the host using Docker

We recommend to use the provided Dockerfile to build a docker image including FINN. It is the easiest way to reproduce 
our results. It is based on the official [FINN installation method](https://finn.readthedocs.io/en/latest/getting_started.html).

## 1.1. Build the docker image

Command to build the docker image (to be run at the root of the cloned repository):
```shell
docker build setup/build -t pose_finn:latest
```

With Podman:
```shell
podman build setup/build -t pose_finn:latest --build-arg VITIS_VERSION=2022.2 --build-arg XILINX_PATH=/CMC/tools/xilinx/Vitis_2022.2/ 
```

| Argument        | Default Value                               | Description                                                                                              |
|-----------------|---------------------------------------------|----------------------------------------------------------------------------------------------------------|
| `VITIS_VERSION` | `2022.1`                                    | Specifies the version of Vitis to be used. It must match the version installed outside.                  |
| `XILINX_PATH`   | `/tools/Xilinx`                             | The directory path inside the container where Vitis is installed. Must be the same as the external path. |
| `FINN_PATH`     | `/tools/finn`                               | The directory path inside the container where FINN will be installed.                                    |
| `PROJECT_ROOT`  | `/workspace/pose_estimation/FPGA-SpacePose` | The directory path inside the container where the project will be mounted.                               |
| `FINN_VERSION`  | `v0.9`                                      | Specifies the release version of FINN from its GitHub releases to be used in the project.                |


## 1.2. Run the docker image

Command to run the docker image (will create an interactive docker container):
```shell
docker run --rm -it --net=host --ipc=host --hostname=finn_pose -v /tools/Xilinx:/tools/Xilinx -v $POSE_ESTIMATION_ROOT:/workspace/pose_estimation pose_finn:latest
```
The Xilinx install directory must be the same on the host and on the container (e.g. /tools/Xilinx)
Replace `$POSE_ESTIMATION_ROOT` with your pose_estimation path that contains both the pose_estimation project and 
the dataset.

```shell
podman run --rm -it --ulimit=host --net=host --ipc=host --hostname=finn_pose -v /CMC/tools/xilinx/Vitis_2022.2/:/CMC/tools/xilinx/Vitis_2022.2/ -v /export/tmp/posso/recherche:/workspace/pose_estimation pose_finn:latest
```

The docker configuration is made so that after running this command, you are on the root of the cloned repository 
(mounted inside the docker container). When the Docker container is initiated, it automatically executes the 
`finn_entrypoint.sh` script.


## 1.3. Verify installation

Verify the installation inside the docker container (should see 1285 passed, 256 skipped, 4 xfailed, 1 xpassed):
```shell
pip install -e git+https://github.com/fbcotter/dataset_loading.git@0.0.4#egg=dataset_loading
bash /tools/finn/docker/quicktest.sh
```


# 2. Installing FINN on the target FPGA-SoC board

## 2.1. Install Pynq on the target 

The following guide is based on the [Getting started page of the Pynq website](https://pynq.readthedocs.io/en/v2.7.0/getting_started.html).

You need to install [Pynq](http://www.pynq.io/) on the board in order to run FINN. 
To reproduce our results, choose Pynq 2.6 for the ZCU104 in the boards section. Other versions of Pynq are available on 
the [GitHub page of Pynq](https://github.com/Xilinx/PYNQ/releases).
We recommend using [Balena Etcher](https://www.balena.io/etcher/) to flash the Pynq image on the MicroSD card. 
If the AppImage of Balena Etcher does not launch correctly, try to add --no-sandbox. For example:

```bash
./balenaEtcher.AppImage --no-sandbox
```


## 2.2. Pynq First time setup 

Once the SD card is ready, insert it into the ZCU104 board, configure the DIP switch according to the image below, 
and follow the next section for the network configuration. 

![ZCU104 configuration](https://pynq.readthedocs.io/en/v2.7.0/_images/zcu104_setup.png)


### 2.2.1. Network configuration

The ZCU104 must have an internet access. The easiest setup is to connect the ZCU104 board to the same 
ethernet switch/router your computer is connected to. 
In that situation, the ZCU104 should have internet access automatically. 
If not working, you may need to [assign a static IP address to your computer](https://linuxconfig.org/how-to-configure-static-ip-address-on-ubuntu-18-10-cosmic-cuttlefish-linux).
If your organization does MAC address filtering, this configuration may not work. Follow the next paragraph.

Alternative setup: directly connect the ZCU104 to your computer with an ethernet cable. 
Thus, you can access the ZCU104 through SSH but the ZCU104 will not have internet access. 
To handle that, open a terminal and nun `nm-connection-editor` and double click to the ethernet interface associated 
with the ZCU104 (should be the most recent one). In the IPv4 section, select "shared to other computers" as method, 
and add an IP address in the range 192.168.2.1/24 except 192.168.2.99 which is the IP of the 
board (e.g. IP=192.168.2.10, MASK=255.255.255.0). Reboot both your computer and the ZCU104.

Make sure the ZCU104 is powered on. Try to connect to the ZCU104 through ssh: `ssh xilinx@192.168.2.99`. 
Once connected to the board, make sure you can access internet: `ping 8.8.8.8`.  


### 2.2.2. SSH the ZCU104 without password

To run FINN we need to be able to connect the ZCU104 through SSH without entering a password. To do so, 
in the dockerfile we generated an RSA public/private key pair in the `/root/.ssh` folder of the docker container. 
We now need to copy the public key to the ZCU104. To do so, follow the next steps: 

Launch a docker container:
```bash
docker run --rm -it --net host --hostname finn_pose -v /tools/Xilinx:/tools/Xilinx -v $POSE_ESTIMATION_ROOT:/workspace/pose_estimation pose_finn:latest
```
Replace `$XILINX_DIR` with the Xilinx Vitis installation directory on your host (e.g. /tools/Xilinx)
Replace `$SPOC_FINN_DIR` with the directory where you cloned the git repo (e.g. /home/julien/irt/spoc_finn)

In the docker container, run the following command:
```bash
ssh-copy-id -i /root/.ssh/id_rsa.pub xilinx@192.168.2.99
```

Verify that you can SSH the ZCU104 without entering a password (will only work in the docker container):
```bash
ssh xilinx@192.168.2.99
```

### 2.2.3. Install packaged on the ZCU104

```bash
sudo pip3 install bitstring
```

