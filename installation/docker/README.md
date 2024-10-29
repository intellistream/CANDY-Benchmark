# Setting up a Remote Docker Environment for CLion IDE

This guide will help developers set up a Docker container on a remote server to allow CLion to connect and work
seamlessly with the containerized environment. The setup involves running a Docker container with SSH capabilities so
that the CLion IDE, running locally, can connect to the container for remote development. The remote container is
CUDA-enabled to support GPU-accelerated development.

## A quick start glance

Go to Setup.pdf

## Prerequisites

You need to have docker installed, if not, try the following for native ubuntu

```bash
cd docker
./DockerOnNativeUbuntu.sh # for native ubuntu, the docker installed in this way requires sudo to run
```

For windows, please see Setup.pdf

### Step 1: Clone the Repository

Start by cloning the repository that contains the `docker-compose.yml`, `Dockerfile`, and the start script.

```bash
# Clone the repository
$ git clone <repository-url>
$ cd <repository-directory>
```

### Step 2: Update the Dockerfile

```bash
cd docker
```

Ensure the Dockerfile is configured to allow SSH access. A root password should be securely set to prevent unauthorized
access.

You can modify the password in the Dockerfile:

#### For the root user

```dockerfile
RUN echo 'root:YourStrongPasswordHere' | chpasswd
```

Replace `YourStrongPasswordHere` with a strong, secure password.

> it is set to root:root by default, i.e., username: root; password: root.

#### For the sudo user

```dockerfile
ENV USER=candy
ENV PASSWD=candy
```

You can just change these marcos for user name and password

### Step 3: Update the Docker Compose File

Ensure the `docker-compose.yml` is correctly set to build and run the Docker image, exposing the SSH port (default is
`2222` in the example).

**docker-compose.yml**:

```yaml
version: '3.8'

services:
  candy:
    build: ./
    volumes:
      - "..:/workspace"
    working_dir: /workspace
    stdin_open: true
    tty: true
    ports:
      - "2222:22"  # Expose SSH port for remote connection
```

Ensure the directory to be mounted (`..:/workspace`) matches your project structure and is correctly set.

### Step 4: Build and Run the Docker Container

Use the provided `start.sh` script to build and run the Docker container.

**Run the following commands**:

```bash
# Make the start script executable
# Run the start script
$ bash start.sh
```

This script will:

1. Stop and remove any previous instances of the Docker container.
2. Build a new container without using the cache.
3. Start the container in detached mode.
   If throw some errors of denney access, use ` sudo bash start.sh ` instead.

### Step 5: Configure SSH for Remote Development

Once the container is up, it will have SSH enabled and listening on port `2222`. You can connect to it from your local
IDE.

1. **Get the IP Address of the Remote Server**:
   Ensure you have the IP address of the remote server where the container is running.

2. **Test the SSH Connection**:
   To verify SSH is working, run the following command:
   ```bash
   ssh root@<remote_server_ip> -p 2222
   ```
   or
   ```bash
   ssh candy@<remote_server_ip> -p 2222
   ```
   Replace `<remote_server_ip>` with the IP address of the remote server. Use the root password set in the Dockerfile,
   it should be 127.0.0.1 if everything is local

### Step 6: Set Up CLion for Remote Development

1. **Open CLion**.

2. **Configure Toolchains**:
    - Go to `File > Settings > Build, Execution, Deployment > Toolchains`.
    - Click `+` to add a new toolchain and choose **Remote Host**.
    - Fill in the SSH connection details:
        - **Host**: `<remote_server_ip>`
        - **Port**: `2222`
        - **Username**: `candy`
        - **Password**: Use the password set in Dockerfile.
          Please just use this sudo user rather than root user, as no root is required by candy

3. **Configure CMake**:
    - Go to `File > Settings > Build, Execution, Deployment > CMake`.
    - Select the newly created toolchain under **Toolchain**.
    - Set **CMake options** if needed for CUDA or specific build configurations.

4. **Sync Project**:
    - There is an automatic mapping of tmp folder, but it is ram-like and only valid when your docker is on.
    - If you want make it consistently work, go to the ` Settings >Deployment `, chose the current machine, and change
      the mappings into /home/candy/candy.
    - The `/home/candy/candy` directory in the container should mirror your project directory on the host machine.

### Step 7: Verify Configuration

After the setup is complete, verify that CLion is able to:

- Build your project using the remote container.
- Run and debug your application seamlessly within the Docker environment.

You can test the connection by building the project from within CLion and observing the output from the remote
container.

## Troubleshooting Tips

- **SSH Issues**: Ensure that the SSH service is properly running in the container (`/usr/sbin/sshd -D`). You can check
  the logs by running `docker logs <container_id>`.
- **Port Conflicts**: If port `2222` is already in use, update the `docker-compose.yml` to expose a different port.
- **Password Issues**: If you cannot connect via SSH, verify the root password matches what is set in the Dockerfile and
  ensure `PasswordAuthentication yes` is set in `/etc/ssh/sshd_config`.

## Security Considerations

- Avoid using the root user in production setups. Instead, create a non-root user with appropriate permissions.
- Use SSH keys for authentication instead of passwords to enhance security.

## Summary

With this setup, developers can leverage a Docker container on a remote server as a build environment, allowing CLion to
connect and provide a local development experience while using remote resources. This can be especially useful for
GPU-accelerated projects where the remote server has specialized hardware not available on the local machine.

