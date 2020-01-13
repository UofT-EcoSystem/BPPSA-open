# BPPSA-open
The (open-source part of) code to reproduce "BPPSA: Scaling Back-propagation by Parallel Scan Algorithm".

Steps:
1. (Optional) If you are using a fresh Ubuntu 18.04 machine without any package on it, you can check out `install.sh` to automate the dependency installation. (Note: read the warning at the top first!)
2. Open `workflow.sh` with an editor. Adjust `DOCKER_BUILD` and `DOCKER_RUN` based on your version of nvidia-docker. (Optional) Adjust `GPU` based on your GPU model name.
3. Run `sh workflow.sh`. Running the whole script takes around 43 hours. You can (optionally) comment out part of the workflow in `workflow.sh` and choose to run a certain step. The things that can be commented out are enclosed in
```
#### Title. ####
...
################
```
