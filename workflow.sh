#!/bin/bash

set -e

# Note: Only comment out the lines that are inside a
#
# #### Title. ####
# ...
# ################
#
# block.


PROJECT_ROOT=`pwd`

################################################################################
#                                                                              #
# Parameters to customize:                                                     #
#                                                                              #
################################################################################

# Change it to "nvidia-docker image build" if you are using a older version.
DOCKER_BUILD="docker image build"

# Change it to "nvidia-docker run" if you are using a older version.
DOCKER_RUN="docker run --gpus all"

GPU=2080ti # or 2070
DOCKERFILE=2080ti.Dockerfile # or 2070.Dockerfile
EPOCHS_RNN_TRAIN_CURVE=100
EPOCHS_RNN_BENCH=10
EPOCHS_GRU_TRAIN_CURVE=400
EPOCHS_GRU_BENCH=400

################################################################################
#                                                                              #
# Each steps are defined below:                                                #
#                                                                              #
################################################################################

build_docker_image () {
  ${DOCKER_BUILD} -f docker/${DOCKERFILE} -t bppsa:0.1 .
}

download_bitstreams () {
  wget https://zenodo.org/record/3612269/files/datasets.zip?download=1 \
      -O datasets.zip
  unzip datasets.zip
  mv -f ./datasets/* ./
  rmdir ./datasets/
  rm -f datasets.zip
}

download_IRMAS () {
  wget https://zenodo.org/record/1290750/files/IRMAS-TrainingData.zip?download=1 \
      -O IRMAS-TrainingData.zip
  unzip IRMAS-TrainingData.zip
  rm -f IRMAS-TrainingData.zip
  mkdir -p IRMASmfcc_s IRMASmfcc_m IRMASmfcc_l
  ${DOCKER_RUN} -it --rm \
    -v ${PROJECT_ROOT}:${PROJECT_ROOT} \
    bppsa:0.1 \
    /bin/bash -c "cd `pwd` && \
                  python IRMAS_parser.py --data-dir ./IRMAS-TrainingData/ \
                                         --save-dir ./IRMASmfcc_s/ \
                                         --frames s && \
                  python IRMAS_parser.py --data-dir ./IRMAS-TrainingData/ \
                                         --save-dir ./IRMASmfcc_m/ \
                                         --frames m && \
                  python IRMAS_parser.py --data-dir ./IRMAS-TrainingData/ \
                                         --save-dir ./IRMASmfcc_l/ \
                                         --frames l"
  rm -rf IRMAS-TrainingData
}

run_rnn_benchmarks () {
  ${DOCKER_RUN} -it --rm \
    -v ${PROJECT_ROOT}:${PROJECT_ROOT} \
    bppsa:0.1 \
    /bin/bash -c "cd `pwd` && python setup.py install && \
                  sh rnn_grid_run.sh ${GPU} \
                                     ${EPOCHS_RNN_TRAIN_CURVE} \
                                     ${EPOCHS_RNN_BENCH}"
}

run_gru_benchmarks () {
  ${DOCKER_RUN} -it --rm \
    -v ${PROJECT_ROOT}:${PROJECT_ROOT} \
    bppsa:0.1 \
    /bin/bash -c "cd `pwd` && python setup.py install && \
                  sh gru_grid_run.sh ${GPU} \
                                     ${EPOCHS_GRU_TRAIN_CURVE} \
                                     ${EPOCHS_GRU_BENCH}"
}

post_process_rnn_gru_results () {
  ${DOCKER_RUN} -it --rm \
    -v ${PROJECT_ROOT}:${PROJECT_ROOT} \
    bppsa:0.1 \
    /bin/bash -c "cd `pwd` && python plot_rnn_results.py --gpu ${GPU} && \
                              python plot_gru_results.py --gpu ${GPU}"
  mv -f ./fig_*.png ../results/
}

run_sparse_jcbT_gen () {
  ${DOCKER_RUN} -it --rm \
    -v ${PROJECT_ROOT}:${PROJECT_ROOT} \
    bppsa:0.1 \
    /bin/bash -c "cd `pwd` && \
                  python jacobian_csr.py > ../results/table_1_last_column.txt"
}

download_csr_jcbTs () {
  wget https://zenodo.org/record/3608306/files/jcbTs.zip?download=1 -O jcbTs.zip
  unzip jcbTs.zip
  rm -f jcbTs.zip
  wget https://zenodo.org/record/3608306/files/jcbTs_prune.zip?download=1 \
      -O jcbTs_prune.zip
  unzip jcbTs_prune.zip
  rm -f jcbTs_prune.zip
}

run_pruned_vgg11_benchmark () {
  ${DOCKER_RUN} -it --rm \
    -v ${PROJECT_ROOT}:${PROJECT_ROOT} \
    bppsa:0.1 \
    /bin/bash -c "cd `pwd` && python pruned_vgg11_analysis.py"
  mv -f ./fig_*.png ../results/
}

################################################################################
#                                                                              #
# Steps to re-produce the results from the paper are listed below:             #
#                                                                              #
################################################################################

steps () {
########################## Build the docker image. #############################
  build_docker_image
################################################################################

  cd ./code/

########################## Download the datasets. ##############################
  download_bitstreams
################################################################################

################ Download and preprocess the IRMAS datasets. ###################
  download_IRMAS
################################################################################

##################### Launch the RNN and GRU experiments. ######################
  run_rnn_benchmarks
  run_gru_benchmarks
################################################################################

  mkdir -p ../results

################## Plot the results for the above experiments. #################
  post_process_rnn_gru_results
################################################################################

####### Produce the speedups for sparse transposed Jacobian Generation. ########
  run_sparse_jcbT_gen
################################################################################


######################### Download the Jacobians. ##############################
  download_csr_jcbTs
################################################################################

######### Produce the results for the pruned VGG-11 micro-benchmark. ###########
  run_pruned_vgg11_benchmark
################################################################################
}

time steps
