#!/bin/bash

# Install Homebrew if it's not installed
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install Docker
echo "Installing Docker..."
brew install --cask docker

# Start Docker Desktop
echo "Starting Docker Desktop..."
open /Applications/Docker.app

# Wait for Docker to start
echo "Waiting for Docker to start..."
while ! docker info >/dev/null 2>&1; do
    sleep 2
done
echo "Docker is up and running."

# Install Docker Compose if needed
if ! command -v docker-compose &> /dev/null; then
    echo "Installing Docker Compose..."
    brew install docker-compose
fi

# Link Docker Compose if not linked
if ! brew list --cask | grep -q docker-compose; then
    echo "Linking Docker Compose..."
    brew link docker-compose
fi

echo "Docker and Docker Compose installation completed successfully."
