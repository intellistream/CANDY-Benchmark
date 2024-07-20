#!/bin/bash
# configure script
rm -rf build
# Enable shared library building
CONFIG_SHARED="--with-shared"

# Pass additional flags to the configure script
./configure ${CONFIG_SHARED} "$@"
# Step 2: Determine the maximum number of threads
max_threads=$(nproc)

make -j${max_threads}
rm build/lib/*.a
