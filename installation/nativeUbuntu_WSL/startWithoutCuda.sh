#!/bin/bash
echo "First, make sure you have sudo"
sudo ls
echo "Installing others..."
sudo apt install -y liblapack-dev libblas-dev
sudo apt-get install -y graphviz libboost-all-dev swig
pip install matplotlib pandas==2.0.0
pip install torch --index-url https://download.pytorch.org/whl/cpu