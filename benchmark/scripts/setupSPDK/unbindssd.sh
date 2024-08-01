#!/bin/bash

NVME_DEVICE="/dev/nvme1n1"
PCI_ADDRESS=""

# Function to unmount the NVMe device
unmount_device() {
    local device=$1
    local mount_point=$(lsblk -no MOUNTPOINT $device)

    if [ -n "$mount_point" ]; then
        echo "Unmounting $device from $mount_point"
        sudo umount $device
        if [ $? -ne 0 ]; then
            echo "Failed to unmount $device"
            exit 1
        fi
    else
        echo "$device is not mounted"
    fi
}

# Function to find the PCI address of the NVMe device
find_pci_address() {
    local device=$1
    local pci_addr=$(ls -l /sys/block/ | grep $(basename $device) | awk '{print $11}' | sed 's/.*\/\(.*\)\/nvme\/.*$/\1/')

    if [ -z "$pci_addr" ]; then
        echo "Failed to find PCI address for $device"
        exit 1
    fi

    echo $pci_addr
}

# Function to unbind the NVMe device from the kernel driver
unbind_nvme() {
    local pci_addr=$1
    echo $pci_addr | sudo tee /sys/bus/pci/devices/$pci_addr/driver/unbind
}

# Function to bind the NVMe device to the uio_pci_generic driver
bind_to_uio_pci_generic() {
    local pci_addr=$1

    # Load the uio and uio_pci_generic modules
    sudo modprobe uio
    sudo modprobe uio_pci_generic

    echo $pci_addr | sudo tee /sys/bus/pci/drivers/uio_pci_generic/bind
}

# Function to expose UIO resources to user space
expose_uio_resources() {
    # Find the UIO device corresponding to the NVMe PCI address
    local uio_device=$(ls /sys/class/uio | head -n 1)

    if [ -z "$uio_device" ]; then
        echo "Failed to find UIO device"
        exit 1
    fi

    # Set permissions for the UIO device
    sudo chmod 666 /dev/$uio_device
    sudo chmod 666 /sys/class/uio/$uio_device/device/config
    sudo chmod 666 /sys/class/uio/$uio_device/device/resource*
}

# Setup hugepages and permissions
setup_hugepages() {
    sudo mkdir -p /mnt/huge
    sudo mount -t hugetlbfs nodev /mnt/huge
    sudo chmod 777 /mnt/huge

    if ! grep -q "hugepagesz=2M hugepages=1024" /etc/default/grub; then
        echo "Updating GRUB configuration for hugepages"
        sudo bash -c 'echo "GRUB_CMDLINE_LINUX_DEFAULT=\"$GRUB_CMDLINE_LINUX_DEFAULT hugepagesz=2M hugepages=1024\"" >> /etc/default/grub'
        sudo update-grub
        echo "Hugepages setup complete. Please reboot the machine for changes to take effect."
        exit 0
    else
        echo "Hugepages are already configured."
    fi
}

# Ensure DPDK uses IOVA as PA
setup_dpdk_config() {
    rm -rf ~/.config/spdk
    mkdir  ~/.config/spdk
    echo "[Global]" > ~/.config/spdk/spdk.conf
    echo "ReactorMask=0x1" >> ~/.config/spdk/spdk.conf
    echo "[Memory]" >> ~/.config/spdk/spdk.conf
    echo "IoVas=PA" >> ~/.config/spdk/spdk.conf
}

# Configure udev rules
setup_udev_rules() {
    echo 'KERNEL=="nvme[0-9]*", SUBSYSTEM=="block", MODE="0666"' | sudo tee /etc/udev/rules.d/99-spdk-nvme.rules
    echo 'SUBSYSTEM=="uio", DEVNAME=="uio*", MODE="0666"' | sudo tee -a /etc/udev/rules.d/99-spdk-nvme.rules

    sudo udevadm control --reload-rules
    sudo udevadm trigger
}

# Check if the NVMe device exists
if [ ! -b $NVME_DEVICE ]; then
    echo "$NVME_DEVICE does not exist"
    #exit 1
fi

# Unmount the NVMe device
unmount_device $NVME_DEVICE

# Find the PCI address of the NVMe device
PCI_ADDRESS=$(find_pci_address $NVME_DEVICE)
echo "Found PCI address: $PCI_ADDRESS"

# Unbind the NVMe device from the kernel driver
unbind_nvme $PCI_ADDRESS

# Bind the NVMe device to the uio_pci_generic driver
bind_to_uio_pci_generic $PCI_ADDRESS

# Expose UIO resources to user space
expose_uio_resources

# Setup hugepages and DPDK configuration
if [ -d "/mnt/huge" ] && [ -w "/mnt/huge" ] && [ -x "/mnt/huge" ]; then
    echo "/mnt/huge is already accessible to the user."
else
    setup_hugepages
fi

# Setup udev rules
setup_udev_rules

# Setup DPDK configuration to use IOVA as PA
setup_dpdk_config
sudo chmod -R 777 /sys/bus/pci/
echo "Setup complete. $NVME_DEVICE is now accessible to SPDK without sudo."
