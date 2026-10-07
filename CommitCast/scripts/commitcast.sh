#!/usr/bin/env bash
set -euo pipefail

python main.py \
  STREAM.ROOT streams \
  RESULT_DIR results/commitcast
