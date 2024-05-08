#!/usr/bin/env python3
import csv
import numpy as np
import os

import os
import pandas as pd
import sys


def getTarGzSingle(url, gzPath, targetPath, fname):
    tryDownload = 0

    fileId = url
    downloadCmd = "cd " + gzPath + "&& wget -O" + fname + " " + fileId

    if os.path.exists(gzPath + "/" + fname):
        return 1
    while tryDownload < 10:
        os.system(downloadCmd)
        os.system('tar -C' + targetPath + ' -zxvf ' + gzPath + "/" + fname)
        if os.path.exists(gzPath + "/" + fname):
            return 1
        tryDownload = tryDownload + 1
    return 0


def getTarGzList(urls, gzPaths, targetPaths, fnames):
    for i in range(len(urls)):
        print(urls[i])
        getTarGzSingle(urls[i], gzPaths[i], targetPaths[i], fnames[i])


def mkdirList(paths):
    for i in range(len(paths)):
        os.system('mkdir ' + paths[i])


def main():
    exeSpace = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/"
    gzPath = exeSpace + 'gzDownload'
    os.system('mkdir ' + gzPath)
    targetPathBase = exeSpace + 'datasets/fvecs'
    os.system('mkdir ' + targetPathBase)
    urls = ['https://github.com/TileDB-Inc/TileDB-Vector-Search/releases/download/0.0.1/siftsmall.tgz',
            'https://figshare.com/ndownloader/files/13755344']
    gzPaths = [gzPath, gzPath]
    targetPaths = [targetPathBase + '/sift10K/',
                   targetPathBase + '/sift1M/']
    fnames = ['siftsmall.tar.gz',
              'sift1M.tar.gz']
    mkdirList(targetPaths)
    getTarGzList(urls, gzPaths, targetPaths, fnames)


if __name__ == "__main__":
    main()
