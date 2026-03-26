#!/bin/bash
set -e

echo "Installing dependencies..."
pip install -r requirements.txt -q

echo "Downloading OpenVaccine dataset from Kaggle..."
mkdir -p data
kaggle competitions download -c stanford-covid-vaccine -p data/
cd data && unzip -o stanford-covid-vaccine.zip && cd ..

echo ""
echo "Data ready in data/"
ls data/
