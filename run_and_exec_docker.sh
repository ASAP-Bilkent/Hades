#!/bin/bash

# Verify the current directory
EXPECTED_DIR="hadesfl"
CURRENT_DIR=$(basename "$PWD")

if [ "$CURRENT_DIR" != "$EXPECTED_DIR" ]; then
  echo "This script must be run from the '$EXPECTED_DIR' directory."
  exit 1
fi

DOCKER_IMAGE="hadesfl"
DOCKERFILE_NAME="Dockerfile"
HOST_PORT=8888
CONTAINER_PORT=8888
HOST_PATH="$PWD"
CONTAINER_PATH="/workspace"

# Check if image exists
if [[ "$(docker images -q ${DOCKER_IMAGE} 2> /dev/null)" == "" ]]; then
  echo "Docker image '${DOCKER_IMAGE}' not found. Building from Dockerfile..."
  docker build -t ${DOCKER_IMAGE} -f ${DOCKERFILE_NAME} .
  if [ $? -ne 0 ]; then
    echo "Docker image build failed."
    exit 1
  fi
else
  echo "Docker image '${DOCKER_IMAGE}' already exists."
fi

# Run the container in detached mode with volume and port mapping
CONTAINER_ID=$(docker run -d -p ${HOST_PORT}:${CONTAINER_PORT} -v "${HOST_PATH}:${CONTAINER_PATH}" ${DOCKER_IMAGE})

if [ -z "$CONTAINER_ID" ]; then
  echo "Failed to start the Docker container."
  exit 1
fi

echo "Docker container started with ID: $CONTAINER_ID"

# Open an interactive shell inside the container
docker exec -it $CONTAINER_ID /bin/bash
