# Build and run the Docker container
docker-compose down --remove-orphans
docker-compose build --no-cache
docker-compose up -d

# Display SSH connection information
echo "Docker container is running. You can connect via SSH with:"
echo "ssh root@<remote_server_ip> -p 2222"

# Get the container ID or name
container_name="docker_candy-1"  # Default container name; adjust if it differs in your case

# Wait for the container to start
sleep 5  # Wait to ensure that container is properly up

# Automatically attach to the container's bash shell
docker exec -it $container_name /bin/bash