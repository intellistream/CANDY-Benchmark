#!/usr/bin/env python3
import csv
import numpy as np
import os

import os
import pandas as pd
import sys


def getHDF5Single(url, targetPath, fname):
    tryDownload = 0

    fileId = url
    downloadCmd = "cd " + targetPath + "&& wget -O" + fname + " " + fileId

    if os.path.exists(targetPath + "/" + fname):
        return 1
    while tryDownload < 10:
        os.system(downloadCmd)
        # os.system('tar -C'+targetPath+' -zxvf '+gzPath+"/"+fname)
        if os.path.exists(targetPath + "/" + fname):
            return 1
        tryDownload = tryDownload + 1
    return 0


def getHDF5List(urls, targetPaths, fnames):
    for i in range(len(urls)):
        print(urls[i])
        getHDF5Single(urls[i], targetPaths[i], fnames[i])


def mkdirList(paths):
    for i in range(len(paths)):
        os.system('mkdir ' + paths[i])


def main():
    exeSpace = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/"
    targetPathBase = exeSpace + 'datasets/hdf5'
    os.system('mkdir ' + targetPathBase)
    urls = ['https://www.dropbox.com/s/z56uf5qdmpp6iqo/enron.hdf5',
            'https://www.dropbox.com/s/h8lvtvfbejghi99/sun.hdf5',
            'https://www.dropbox.com/s/9ezi2gkuhnkem6d/trevi.hdf5',
            'https://www.dropbox.com/s/xg0jvdnp8oszhuu/glove.hdf5',
            'https://www.dropbox.com/s/mh11y5q7dugehwi/millionSong.hdf5']
    targetPaths = [targetPathBase + '/enron/',
                   targetPathBase + '/sun/',
                   targetPathBase + '/trevi/',
                   targetPathBase + '/glove/',
                   targetPathBase + '/msong/', ]
    fnames = ['enron.hdf5',
              'sun.hdf5',
              'trevi.hdf5',
              'glove.hdf5',
              'msong.hdf5']
    mkdirList(targetPaths)
    getHDF5List(urls, targetPaths, fnames)


if __name__ == "__main__":
    main()
