# CrazyFlie Control

Simulation for CrazyFlie control algorithms.

Software version: 
- **ROS 2:** Humble [[docs](https://docs.ros.org/en/humble/index.html)]
- **Gazebo:** Harmonic [[docs](https://gazebosim.org/docs/harmonic/getstarted/)]

> [!NOTE]
> This repository is prepared for working with Ubuntu OS. 
> If you are using Windows, please open Ubuntu in WSL 2.

## Prerequisites

First of all, before you start anything, please fork this repository and **work on your own copy**.

The best way to work with this repository is to use the prepared Docker image.
Follow the official [Docker Documentation](https://docs.docker.com/) to install the required software (note that on Linux, only the Docker Engine is required).
Also, if you have an Nvidia GPU and want to use its computing power within Docker, you should install the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (without experimental packages and rootless mode).

> [!NOTE]
> If you do not have an Nvidia GPU card, you need to modify the [devcontainer.json](.devcontainer/devcontainer.json) and set `"dockerComposeFile"` to `compose.nogpu.yaml`.

## Docker setup

> [!TIP]
> In Windows, it is recommended to use WSL2 with a native Linux file system.
> Normally directories from Windows are mounted as `/mnt` inside Ubuntu, but this may cause some problems due to the NTFS file system and permissions.
> Therefore, copy repository or clone it directly into `$HOME` directory.

Open the project's directory in VS Code.
In the bottom left corner, click on the icon with the two arrows pointing at each other.
The menu with different options will appear at the top - choose **"Open folder in container... "** and wait, the docker will be created.
Note that this may take some time (depending on your internet connection).

In terminal inside docker container, run:

```bash
cd /home/developer/ros2_ws
source ./setup.sh 23
./build.sh
source install/setup.bash
cd src
```

The `<id>` will be set as `ROS_DOMAIN_ID` in the Docker environment. 
In order not to interfere with the work of other colleagues in the laboratory, it is necessary to have a **unique identifier**!

> [!NOTE]
> In Windows, there may be a problem with running bash scripts - two possible reasons are:
>
> 1. Scripts are not set to be executable files - you can change it with `sudo chmod +x script.sh`.
> 2. Wrong line end character - change `CRLF` to `LF` in the bottom right part of the VS Code GUI.

Above commands built your workspace with all needed dependencies.

### First run

Run the simulation with:

```bash
ros2 launch ros_gz_crazyflie_bringup crazyflie_simulation.launch.py
```

In separate terminal run the ROS wrapper for the CrazyFlie model:

```bash
ros2 run cf_control mixer
```

Now, in third terminal, when you run `ros2 topic list` you should see the topic `/cf_control/control_command` listed among others.
This is the input control interface for your algorithm - there you should publish control as collective thrust with torque.
You can check if it working by invoking the following command:

```bash
ros2 topic pub /cf_control/control_command cf_control_msgs/msg/ThrustAndTorque "{collective_thrust: 0.295}"
```

Your drone should take off in the simulation 🙂

### Remarks on working with repository

Create a new ROS2 package and fill it with some valuable content :smile:.

When working with the ROS workspace, it is sometimes necessary to rebuild the whole workspace, e.g. if you want to run some script from the newly created package.
You can do this by running the [build.sh](.devcontainer/build.sh) script located in the `ros2_ws` directory inside docker.
Remember to also source the `~/ros2_ws/install/setup.bash` script.
Note that this is added to `~/.bashrc` so that it is automatically sourced when a new terminal is created.

For your convenience, the default ROS log path has been changed to `~/ros2_ws/src/logs`.
This will allow you to access them from outside the container.

### Data exchange between PX4 and ROS

In order to communicate with PX4 from ROS, it is necessary to open the special bridge with [micro_ros_agent](https://github.com/micro-ROS/micro-ROS-Agent):

```bash
ros2 run micro_ros_agent micro_ros_agent udp4 --port 8888
```

Internally, this package uses the Micro XRCE-DDS Agent.
You can read more about topics' bridging in the [PX4 documentation](https://docs.px4.io/v1.15/en/ros2/user_guide.html#setup-micro-xrce-dds-agent-client).

## Additional features

### Data exchange between Gazebo and ROS2

Natively, Gazebo and ROS have a similar internal structure with topics and services to transport data, but they are not seamlessly interoperable (e.g. different types).
However, there is an additional [ros_gz](https://github.com/gazebosim/ros_gz) package provided by Gazebo developers, which facilitates integration between Gazebo and ROS.
It is already installed in the docker, so you can use all the features you need.

### Rosbag

If you are running nodes inside ROS2, you can easily record all the data that is exchanged via topics.
Just use the [rosbag2](https://github.com/ros2/rosbag2) which is already installed in the docker.
Remember that you can manually select the topics you want to record and set the path where all the bags will be stored.

### PlotJuggler

If you've recorded some rosbags, you'd like to be able to read and analyse them now, wouldn't you?
Fortunately, there is a good piece of software called [PlotJuggler](https://plotjuggler.io/) that is already installed in the docker.
You can run it with the following command:

```bash
ros2 run plotjuggler plotjuggler 
```
After some random meme, you will see the main window where you can import and plot data.
Some additional functions can be found in the linked documentation.

## FAQs

### I started the Gazebo, but simulator window did not pop up

Re-open the folder locally and give additional permissions for Docker:
``` bash
xhost +local:docker
```

Re-open the folder in the container and check that it is now working correctly.
