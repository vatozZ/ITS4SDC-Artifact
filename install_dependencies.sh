#!/bin/bash

echo "Trying to install TensorFlow with GPU (tensorflow[and-cuda])..."

if python3 -m pip install "tensorflow[and-cuda]" ; then
    echo "Successfully installed tensorflow[and-cuda]"
else
    echo "Could not install tensorflow[and-cuda], falling back to CPU version..."
    echo "Installing fixed versions from requirements..."

    pip install
fi
