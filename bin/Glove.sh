#!/bin/bash

# Glove data download script

# Input data directory
echo "Enter the data directory path for Glove data: (e.g. /path/to/data/Glove)"
read DATA_DIR

RAW_DATA_DIR=$DATA_DIR/raw
PROCESSED_DATA_DIR=$DATA_DIR/processed

# Create data directories
mkdir -p $RAW_DATA_DIR
mkdir -p $PROCESSED_DATA_DIR

# Function to fetch data
fetch_data() {
    echo "Fetching data..."
    urls=("https://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip" "https://nlp.stanford.edu/data/wordvecs/glove.6B.zip" "https://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip")
    for url in "${urls[@]}"; do
        echo "Fetching data from $url..."
        wget -P $RAW_DATA_DIR $url
    done
    echo "Data fetched successfully!"
}

unzip_data() {
    echo "Unzipping data..."
    for file in $RAW_DATA_DIR/*.zip; do
        echo "Unzipping $file..."
        unzip -d $RAW_DATA_DIR $file
        rm -rf $file
    done
    echo "Data unzipped successfully!"
}

# Main script execution
fetch_data
unzip_data 