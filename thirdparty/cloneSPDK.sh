#!/bin/bash
# configure script
git clone https://github.com/spdk/spdk.git 
cd spdk 
git checkout 94a53a5
cp ../installSPDK.sh .
git submodule update --init
rm -rf app
rm configure
cp -r ../spdk_patch/app .
cp -r ../spdk_patch/configure .
sudo scripts/pkgdep.sh
./installSPDK.sh

