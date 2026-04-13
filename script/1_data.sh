#!/usr/bin/env bash

set -euo pipefail

# Canonical data-preparation commands for the paper reproduction path.
# Run from the repository root after downloading:
#   data/ipcai_2020_full_res_data.zip

python -m src.data.1_extract_content
python -m src.data.2_project --task_type hard
