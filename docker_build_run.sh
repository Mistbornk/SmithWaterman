#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
docker build -t smithwaterman-cuda "$repo_dir"
docker run --gpus all --rm -v "${repo_dir}:/app" smithwaterman-cuda   /bin/bash -c 'cmake -S /app -B /tmp/sw-build -DCMAKE_BUILD_TYPE=Release && cmake --build /tmp/sw-build --parallel 2 && ctest --test-dir /tmp/sw-build --output-on-failure'
