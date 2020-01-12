# To build the docker image.

In the root directory of `BPPSA-open/`, run:

```bash
nvidia-docker image build -f docker/Dockerfile -t bppsa:0.1 .
```

# To run something (a script, executable, etc.).

Assuming the path to `BPPSA-open` is `$HOME/workspace/BPPSA-open/`:

```bash
# Old version of nvidia-docker:
nvidia-docker run -it --rm -v $HOME/workspace/BPPSA-open/:$HOME/workspace/BPPSA-open bppsa:0.1 /bin/bash -c "cd `pwd` && python ..."

# New version of nvidia-docker:
docker run --gpus all -it --rm -v $HOME/workspace/BPPSA-open/:$HOME/workspace/BPPSA-open bppsa:0.1 /bin/bash -c "cd `pwd` && python ..."
```

Note that `-v` mount a directory in the host machine (`$HOME/workspace/BPPSA-open/` in this case) to a path in the docker container (`$HOME/workspace/BPPSA-open/`).
