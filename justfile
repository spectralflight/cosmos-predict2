default:
  just --list

# Setup the repository
setup:
  uv tool install -U pre-commit
  pre-commit install -c .pre-commit-config-base.yaml

# Install the repository
install:
  uv sync

# Run linting and formatting
lint: setup
  pre-commit run --all-files || pre-commit run --all-files

# Run tests
test: lint

# Generate the ATTRIBUTIONS.txt file.
license: install
  uvx licensecheck
  uvx pip-licenses --python .venv/bin/python --format=plain-vertical --with-license-file --no-license-path --no-version --with-urls --output-file ATTRIBUTIONS.txt

# Build the docker image.
docker-build cuda_version='12.6.3' *args:
  docker build --build-arg BASE_IMAGE="nvidia/cuda:{{cuda_version}}-cudnn-devel-ubuntu24.04" -t "cosmos-predict2:{{cuda_version}}" -f uv.Dockerfile {{args}} .

# Run the docker container.
docker cuda_version='12.6.3' *args:
  # https://github.com/astral-sh/uv-docker-example/blob/main/run.sh
  just -f {{justfile()}} docker-build "{{cuda_version}}"
  docker run --gpus all --rm -v .:/workspace -v /workspace/.venv -it "cosmos-predict2:{{cuda_version}}" {{args}}

docker-arm *args:
  docker pull nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2
  docker run --gpus all --rm -v .:/workspace -it "cosmos-predict2:1.2" {{args}}