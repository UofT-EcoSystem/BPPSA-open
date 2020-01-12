set -e

# WARNING!!! : Use this script only if you want to run experiments on a fresh
# *Ubuntu 18.04* machine with a *Nvidia GPU* BUT without the driver, docker-ce
# and nvidia-docker, and also be prepared to wipe the machine if something goes
# wrong. We have not intensively test this script; and since it has to be run
# in sudo mode, use it at your own risk!
#
# The authors are NOT legally resposible for any damage caused by any potential
# malfunction of this script.
#
# YOU HAVE BEEN WARNED!

# Install the driver and CUDA.
dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
apt-get update
apt-get install cuda

# Install docker-ce.
apt-get update
apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
apt-get update
apt-get install docker-ce docker-ce-cli containerd.io
groupadd docker
usermod -aG docker $USER
newgrp docker

# Install nvidia-docker.
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update && apt-get install -y nvidia-container-toolkit
systemctl restart docker
