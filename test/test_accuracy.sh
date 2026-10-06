#!/usr/bin/env bash
# Usage: test/test_accuracy.sh [build-directory] [repeat-count]
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${1:-${script_dir}/../build}"
repeat="${2:-1}"
if ! [[ "$repeat" =~ ^[1-9][0-9]*$ ]]; then
  echo "repeat-count must be a positive integer" >&2
  exit 2
fi
exec ctest --test-dir "$build_dir" --output-on-failure --repeat "until-fail:${repeat}" -L correctness
