set -e

# Change it to "nvidia-docker image build" if you are using a older version.
DOCKER_BUILD=docker image build

# Change it to "nvidia-docker run" if you are using a older version.
DOCKER_RUN=docker run --gpus all

# Build the docker image.
${DOCKER_BUILD} -f docker/Dockerfile -t bppsa:0.1 .

cd ./code/

# Download the datasets.
wget https://zenodo.org/record/3605369/files/datasets.zip?download=1 -O datasets.zip
unzip datasets.zip

${DOCKER_RUN} -it --rm \
  -v $HOME/workspace/BPPSA-open/:$HOME/workspace/BPPSA-open \
  bppsa:0.1 \
  /bin/bash -c "cd `pwd` && python setup.py && sh rnn_grid_run.sh && sh gru_grid_run.sh"
