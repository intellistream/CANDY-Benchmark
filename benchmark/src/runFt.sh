#!/bin/bash

# Check if the number of arguments is greater than 0
if [ "$#" -gt 0 ]; then
    # Call the program a.out with all the input parameters
    export OMP_NUM_THREADS=1
    ./ft "$@"
else
    echo "Usage: $0 [arguments for ft]"
fi