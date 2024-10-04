#!/bin/bash

# Audio data download script

# Input data directory
echo "Enter the data directory path for Audio data: (e.g. /path/to/data/Audio)"
read DATA_DIR

RAW_DATA_DIR=$DATA_DIR/raw
PROCESSED_DATA_DIR=$DATA_DIR/processed

# Create data directories
mkdir -p $RAW_DATA_DIR
mkdir -p $PROCESSED_DATA_DIR

# Function to fetch data
fetch_data() {
    echo "Fetching data..."
    wget -P $RAW_DATA_DIR "http://www.cs.princeton.edu/cass/audio.tar.gz"
    echo "Data fetched successfully!"
}

unzip_data() {
    echo "Unzipping data..."
    tar -xvzf $RAW_DATA_DIR/audio.tar.gz -C $RAW_DATA_DIR
    rm -rf $RAW_DATA_DIR/audio.tar.gz
    echo "Data unzipped successfully!"
}

# Main script execution
fetch_data
unzip_data