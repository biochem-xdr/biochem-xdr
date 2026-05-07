#!/bin/bash
set -e

RESULTS=${1:-results}
OUTPUT=${2:-analysis_outputs}

mkdir -p "$OUTPUT"
python analysis/main_results.py "$RESULTS" "$OUTPUT"
echo "Saved to $OUTPUT/table1_main_results.csv"