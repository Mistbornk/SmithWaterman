#!/bin/bash
set -e

# Build the Docker image
echo "Building Docker image..."
docker build -t smithwaterman-cuda .

# Run the container to build and test
echo "Running build and test inside Docker container..."
docker run --gpus all --rm -v $(pwd):/app smithwaterman-cuda /bin/bash -c "cd build && cmake .. && make && ./test/biovoltron-test"