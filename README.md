# BPPSA-open
The (open-source part of) code to reproduce "BPPSA: Scaling Back-propagation by Parallel Scan Algorithm".

## Dependencies ##
- [`nvidia-docker`](https://github.com/NVIDIA/nvidia-docker)
- `wget` for downloading the datasets. (Note: you don't have to use `wget`, as long as you can get the datasets to be placed under `code/`, i.e., `code/bernoulli{10...30000}_10/` and `code/IRMASmfcc_{s|m|l}/`. If you choose to do so, you can comment out the corresponding part in `workflow.sh` to avoid the downloading time, although the script still works if the datasets are downloaded again.)

## Steps ##
1. (Optional) If you are using a fresh Ubuntu 18.04 machine without any package on it, you can check out `install.sh` to automate the dependency installation. (Note: read the warning at the top first!)
2. Open `workflow.sh` with an editor. Adjust `DOCKER_BUILD` and `DOCKER_RUN` based on your version of nvidia-docker. (Optional) Adjust `GPU` and `DOCKERFILE` based on your GPU model name.
3. Run `./workflow.sh`. Running the whole script takes around 57 hours. You can (optionally) comment out part of the workflow in `workflow.sh` and choose to run a certain step. The things that can be commented out are enclosed in
```
#### Title. ####
...
################
```
4. At the end of the workflow, a `results/` directory will be created that contains figures similar to the ones inside the paper.

### Notes ###
1. By using (including but not limited to: copy, modify, merge, publish, distribute) any part of this project, you consent to the terms in our license.
2. By running `workflow.sh`, you consent to the terms in [IRMAS' license](https://www.upf.edu/web/mtg/irmas).
