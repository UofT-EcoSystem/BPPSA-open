# BPPSA-open
The (open-source part of) code to reproduce "BPPSA: Scaling Back-propagation by Parallel Scan Algorithm".

Steps:
1. (Optional) If you are using a fresh Ubuntu 18.04 machine without any package on it, you can check out `install.sh` to automate the dependency installation. (Note: read the warning at the top first!)
2. Open `workflow.sh` with an editor. Adjusting `DOCKER_BUILD` and `DOCKER_RUN` based on your version of nvidia-docker.
3. Run `sh workflow.sh`.
