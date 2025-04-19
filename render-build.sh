#!/bin/bash
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Checking installed packages:"
pip list