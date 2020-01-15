set -e

# Note: Only comment out the lines that are inside a
#
# #### Title. ####
# ...
# ################
#
# block.

# The model of the GPU being used.
GPU=2080ti # 2070

# Change it to "nvidia-docker image build" if you are using a older version.
DOCKER_BUILD="docker image build"

# Change it to "nvidia-docker run" if you are using a older version.
DOCKER_RUN="docker run --gpus all"

PROJECT_ROOT=`pwd`

########################## Build the docker image. #############################
${DOCKER_BUILD} -f docker/Dockerfile -t bppsa:0.1 .
################################################################################

cd ./code/

########################## Download the datasets. ##############################
wget https://zenodo.org/record/3605369/files/datasets.zip?download=1 \
    -O datasets.zip
unzip datasets.zip
mv -f ./dataset/* ./
rmdir ./dataset/
################################################################################

##################### Launch the RNN and GRU experiments. ######################
${DOCKER_RUN} -it --rm \
  -v ${PROJECT_ROOT}:${PROJECT_ROOT} \
  bppsa:0.1 \
  /bin/bash -c "cd `pwd` && python setup.py install && \
                sh rnn_grid_run.sh ${GPU} && \
                sh gru_grid_run.sh ${GPU}"
################################################################################

mkdir -p ../results

################## Plot the results for the above experiments. #################
${DOCKER_RUN} -it --rm \
  -v ${PROJECT_ROOT}:${PROJECT_ROOT} \
  bppsa:0.1 \
  /bin/bash -c "cd `pwd` && python plot_rnn_results.py --gpu ${GPU} && \
                            python plot_gru_results.py --gpu ${GPU}"
mv -f ./fig_*.png ../results/
################################################################################

####### Produce the speedups for sparse transposed Jacobian Generation. ########
${DOCKER_RUN} -it --rm \
  -v ${PROJECT_ROOT}:${PROJECT_ROOT} \
  bppsa:0.1 \
  /bin/bash -c "cd `pwd` && \
                python jacobian_csr.py > ../results/table_1_last_column.txt"
################################################################################


######################### Download the Jacobians. ##############################
wget https://zenodo.org/record/3608306/files/jcbTs.zip?download=1 -O jcbTs.zip
unzip jcbTs.zip
wget https://zenodo.org/record/3608306/files/jcbTs_prune.zip?download=1 \
    -O jcbTs_prune.zip
unzip jcbTs_prune.zip
################################################################################


######### Produce the results for the pruned VGG-11 micro-benchmark. ###########
${DOCKER_RUN} -it --rm \
  -v ${PROJECT_ROOT}:${PROJECT_ROOT} \
  bppsa:0.1 \
  /bin/bash -c "cd `pwd` && \
                python pruned_vgg11_analysis.py"
mv -f ./fig_*.png ../results/
################################################################################
