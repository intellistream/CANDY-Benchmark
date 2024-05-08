#!/usr/bin/env python3
import csv
import numpy as np
import os

import os
import pandas as pd
import sys
import downLoadTarGz
import downLoadHDF5


def main():
    downLoadTarGz.main()
    downLoadHDF5.main()


if __name__ == "__main__":
    main()
