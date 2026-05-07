#!/bin/bash
set -e

RESULTS=${1:-results}
DATASET=${2:-data/biochem_xdr_full.json}
TIER_FILE=${3:-configs/tier_aggregate_calibrated.csv}
FIGURES=${4:-figures_output}

mkdir -p "$FIGURES"
python figures/generate_all_figures.py "$RESULTS" "$DATASET" "$TIER_FILE" "$FIGURES"
echo "Saved to $FIGURES"