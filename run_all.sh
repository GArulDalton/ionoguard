#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/scripts"
python train_models.py && python run_main.py && python run_sweeps.py && python run_trace.py \
  && python timing.py && python summarize.py && python make_figures.py && python make_block_diagrams.py
