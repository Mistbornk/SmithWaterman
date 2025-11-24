FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Default command
CMD ["/bin/bash"]
