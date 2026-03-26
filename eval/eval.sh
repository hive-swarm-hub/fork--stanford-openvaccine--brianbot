#!/bin/bash
set -e

echo "=== Training model ==="
python train.py

echo ""
echo "=== Computing score ==="
python eval/score.py
