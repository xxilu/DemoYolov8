#!/bin/bash

# Create Environemt
echo "Activating conda environment..."
conda create -n yolov8-demo python=3.12 -y
conda activate yolov8-demo

# Installation
echo "Installation..."
pip install -r requirement.txt

# Run Flask
echo "Starting Flask server..."
python app.py
