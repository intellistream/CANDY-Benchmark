#!/usr/bin/env python3
# Note: the concept drift is not learnt by indexing in this group
import csv
import numpy as np
import matplotlib.pyplot as plt
import accuBar as accuBar
import groupBar2 as groupBar2
import groupLine as groupLine
from autoParase import *
import itertools as it
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pylab
from matplotlib.font_manager import FontProperties
from matplotlib import ticker
from matplotlib.ticker import LogLocator, LinearLocator

import os
import pandas as pd
import sys
from OoOCommon import *

OPT_FONT_NAME = 'Helvetica'
TICK_FONT_SIZE = 22
LABEL_FONT_SIZE = 22
LEGEND_FONT_SIZE = 22
LABEL_FP = FontProperties(style='normal', size=LABEL_FONT_SIZE)
LEGEND_FP = FontProperties(style='normal', size=LEGEND_FONT_SIZE)
TICK_FP = FontProperties(style='normal', size=TICK_FONT_SIZE)

MARKERS = (['*', '|', 'v', "^", "", "h", "<", ">", "+", "d", "<", "|", "", "+", "_"])
# you may want to change the color map for different figures
COLOR_MAP = (
    '#B03A2E', '#2874A6', '#239B56', '#7D3C98', '#FFFFFF', '#F1C40F', '#F5CBA7', '#82E0AA', '#AEB6BF', '#AA4499')
# you may want to change the patterns for different figures
PATTERNS = (["////", "o", "", "||", "-", "//", "\\", "o", "O", "////", ".", "|||", "o", "---", "+", "\\\\", "*"])
LABEL_WEIGHT = 'bold'
LINE_COLORS = COLOR_MAP
LINE_WIDTH = 3.0
MARKER_SIZE = 15.0
MARKER_FREQUENCY = 1000

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['xtick.labelsize'] = TICK_FONT_SIZE
matplotlib.rcParams['ytick.labelsize'] = TICK_FONT_SIZE
matplotlib.rcParams['font.family'] = OPT_FONT_NAME
matplotlib.rcParams['pdf.fonttype'] = 42


def runPeriod(exePath, algoTag, resultPath, configTemplate="config.csv", prefixTagRaw="null"):
    # resultFolder="periodTests"
    prefixTag = str(prefixTagRaw)
    configFname = "config_period_" + str(prefixTag) + ".csv"
    configTemplate = "config_e2e_static_lazy.csv"
    # clear old files
    os.system("cd " + exePath + "&& sudo rm *.csv")
    os.system("cp perfListEvaluation.csv " + exePath)
    # editConfig(configTemplate, exePath + configFname, "earlierEmitMs", 0)
    editConfig(configTemplate, "temp2.csv", "eventRateTps", prefixTag)
    editConfig("temp2.csv", exePath + "temp1.csv", "faissIndexTag", algoTag)
    if (algoTag == 'LSH'):
        editConfig(exePath + "temp1.csv", exePath + "temp2.csv", "numberOfBuckets", 1)
        editConfig(exePath + "temp2.csv", exePath + "temp3.csv", "useCRS", 0)
        editConfig(exePath + "temp3.csv", exePath + "temp4.csv", "congestionDropWorker_algoTag", "onlineIVFLSH")
        editConfig(exePath + "temp4.csv", exePath + "temp1.csv", "encodeLen", 3)
    if (algoTag == 'LSH-H'):
        editConfig(exePath + "temp1.csv", exePath + "temp2.csv", "congestionDropWorker_algoTag", "onlineIVFLSH")
        editConfig(exePath + "temp2.csv", exePath + "temp4.csv", "useCRS", 0)
        editConfig(exePath + "temp4.csv", exePath + "temp1.csv", "encodeLen", 3)
    if (algoTag == 'flatAMMIP'):
        editConfig(exePath + "temp1.csv", exePath + "temp2.csv", "congestionDropWorker_algoTag", "flatAMMIP")
        editConfig(exePath + "temp2.csv", exePath + "temp1.csv", "sketchSize", 256)
    if (algoTag == 'flatAMMIPSMPPCA'):
        editConfig(exePath + "temp1.csv", exePath + "temp2.csv", "congestionDropWorker_algoTag", "flatAMMIP")
        editConfig(exePath + "temp2.csv", exePath + "temp4.csv", "sketchSize", 128)
        editConfig(exePath + "temp4.csv", exePath + "temp1.csv", "ammAlgo", 'smp-pca')
    if (algoTag == 'flat'):
        editConfig(exePath + "temp1.csv", exePath + "temp2.csv", "congestionDropWorker_algoTag", "flat")
        editConfig(exePath + "temp2.csv", exePath + "temp1.csv", "sketchSize", 256)
    if (algoTag == 'NSW'):
        editConfig(exePath + "temp1.csv", exePath + "temp2.csv", "congestionDropWorker_algoTag", "NSW")
        editConfig(exePath + "temp2.csv", exePath + "temp1.csv", "is_NSW", 1)
    if (algoTag == 'nnDescent'):
        editConfig(exePath + "temp1.csv", exePath + "temp2.csv", "congestionDropWorker_algoTag", "nnDescent")
        editConfig(exePath + "temp2.csv", exePath + "temp1.csv", "frozenLevel", 1)
    if (algoTag == 'onlinePQ'):
        editConfig(exePath + "temp1.csv", exePath + "temp3.csv", "faissIndexTag", "PQ")
        editConfig(exePath + "temp3.csv", exePath + "temp2.csv", "isOnlinePQ", 1)
        editConfig(exePath + "temp2.csv", exePath + "temp1.csv", "sketchSize", 256)
    if (algoTag == 'Flann'):
        editConfig(exePath + "temp1.csv", exePath + "temp2.csv", "congestionDropWorker_algoTag", "Flann")
        editConfig(exePath + "temp2.csv", exePath + "temp1.csv", "sketchSize", 256)
    if (algoTag == 'DPG'):
        editConfig(exePath + "temp1.csv", exePath + "temp2.csv", "congestionDropWorker_algoTag", "DPG")
        editConfig(exePath + "temp2.csv", exePath + "temp1.csv", "frozenLevel", 1)
    exeTag = "onlineInsert"
    # prepare new file
    os.system("rm -rf " + exePath + "*.rbt")
    os.system("cp *.rbt " + exePath)
    # run
    os.system("cd " + exePath + "&& export OMP_NUM_THREADS=1 &&" + "sudo ./" + exeTag + " " + 'temp1.csv')
    # copy result
    os.system("sudo rm -rf " + resultPath + "/" + str(prefixTag))
    os.system("sudo mkdir " + resultPath + "/" + str(prefixTag))

    os.system("cd " + exePath + "&& sudo cp *.csv " + resultPath + "/" + str(prefixTag))


def runPeriodVector(exePath, algoTag, resultPath, prefixTag, configTemplate="config.csv", reRun=1):
    for i in range(len(prefixTag)):
        if reRun == 2:
            if checkResultSingle(prefixTag[i], resultPath) == 1:
                print("skip " + str(prefixTag[i]))
            else:
                runPeriod(exePath, algoTag, resultPath, configTemplate, prefixTag[i])
        else:
            runPeriod(exePath, algoTag, resultPath, configTemplate, prefixTag[i])


def readResultSingle(singleValue, resultPath):
    resultFname = resultPath + "/" + str(singleValue) + "/onlineInsert_result.csv"
    elapsedTime = readConfig(resultFname, "latencyOfQuery")
    incrementalBuild = readConfig(resultFname, "95%latency(Insert)")
    incrementalSearch = readConfig(resultFname, "latencyOfQuery")
    recall = readConfig(resultFname, "recall")
    pendingWaitTime = readConfig(resultFname, "pendingWrite")
    l2Stall = 0
    l3Stall = 0
    totalStall = 0
    froErr = 0
    return elapsedTime, incrementalBuild, incrementalSearch, recall, pendingWaitTime, l2Stall, l3Stall, totalStall, froErr


def readResultVector(singleValueVec, resultPath):
    elapseTimeVec = []
    incrementalBuildVec = []
    incrementalSearchVec = []
    recallVec = []
    pendingWaitTimeVec = []
    l2StallVec = []
    l3StallVec = []
    totalStallVec = []
    froVec = []
    for i in singleValueVec:
        elapsedTime, incrementalBuild, incrementalSearch, recall, pendingWaitTime, l2Stall, l3Stall, totalStall, fro = readResultSingle(
            i, resultPath)
        elapseTimeVec.append(float(elapsedTime))
        incrementalBuildVec.append(float(incrementalBuild))
        incrementalSearchVec.append(float(incrementalSearch))
        recallVec.append(float(recall))
        pendingWaitTimeVec.append(float(pendingWaitTime))
        l2StallVec.append(float(l2Stall))
        l3StallVec.append(float(l3Stall))
        totalStallVec.append(float(totalStall))
        froVec.append(float(fro))
    return np.array(elapseTimeVec), np.array(incrementalBuildVec), np.array(incrementalSearchVec), np.array(
        recallVec), np.array(
        pendingWaitTimeVec), np.array(l2StallVec), np.array(l3StallVec), np.array(totalStallVec), np.array(froVec)


def checkResultSingle(singleValue, resultPath):
    resultFname = resultPath + "/" + str(singleValue) + "/onlineInsert_result.csv"
    ruExists = 0
    if os.path.exists(resultFname):
        ruExists = 1
    else:
        print("File does not exist:" + resultFname)
        ruExists = 0
    return ruExists


def checkResultVector(singleValueVec, resultPath):
    resultIsComplete = 0
    for i in singleValueVec:
        resultIsComplete = checkResultSingle(i, resultPath)
        if resultIsComplete == 0:
            return 0
    return 1


def compareMethod(exeSpace, commonPathBase, resultPaths, csvTemplate, algos, dataSetName, reRun=1):
    elapsedTimeAll = []
    incrementalBuildAll = []
    incrementalSearchAll = []
    periodAll = []
    recallAll = []
    pendingWaitTimeAll = []
    l2StallAll = []
    l3StallAll = []
    totalStallAll = []
    froAll = []
    resultIsComplete = 1
    algoCnt = 0
    for i in range(len(algos)):
        resultPath = commonPathBase + resultPaths[i]
        algoTag = algos[i]
        scanVec = dataSetName
        if (reRun == 1):
            os.system("sudo rm -rf " + resultPath)
            os.system("sudo mkdir " + resultPath)
            runPeriodVector(exeSpace, algoTag, resultPath, scanVec, csvTemplate)
        else:
            if (reRun == 2):
                resultIsComplete = checkResultVector(scanVec, resultPath)
                if resultIsComplete == 1:
                    print(algoTag + " is complete, skip")
                else:
                    print(algoTag + " is incomplete, redo it")
                    if os.path.exists(resultPath) == False:
                        os.system("sudo mkdir " + resultPath)
                    runPeriodVector(exeSpace, algoTag, resultPath, scanVec, csvTemplate, 2)
                    resultIsComplete = checkResultVector(scanVec, resultPath)
        # exit()
        if resultIsComplete:
            elapsedTime, incrementalBuild, incrementalSearch, recall, pendingWaitTime, l2Stall, l3Stall, totalStall, froVec = readResultVector(
                dataSetName, resultPath)
            elapsedTimeAll.append(elapsedTime)
            incrementalBuildAll.append(incrementalBuild)
            incrementalSearchAll.append(incrementalSearch)
            periodAll.append(dataSetName)
            recallAll.append(recall)
            pendingWaitTimeAll.append(pendingWaitTime)
            l2StallAll.append(l2Stall)
            l3StallAll.append(l3Stall)
            totalStallAll.append(totalStall)
            froAll.append(froVec)
            algoCnt = algoCnt + 1
            print(algoCnt)
        # periodAll.append(periodVec)
    return np.array(elapsedTimeAll), np.array(incrementalBuildAll), np.array(periodAll), np.array(recallAll), np.array(
        incrementalSearchAll), np.array(pendingWaitTimeAll), np.array(l2StallAll), np.array(l3StallAll), np.array(
        totalStallAll), np.array(froAll)


def getCyclesPerMethod(cyclesAll, valueChose):
    recallPerMethod = []
    for i in range(len(cyclesAll)):
        recallPerMethod.append(cyclesAll[int(i)][int(valueChose)])
    return np.array(recallPerMethod)


def main():
    exeSpace = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/"
    commonBasePath = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/results/scanIPEventRate/"

    figPath = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/figures/scanIPEventRate"

    # add the datasets here
    # srcAVec=["datasets/AST/mcfe.mtx"] # 765*756
    # srcBVec=["datasets/AST/mcfe.mtx"] # 765*756
    # dataSetNames=['AST']
    # srcAVec=['datasets/UTM/utm1700a.mtx'] # 1700*1700
    # srcBVec=['datasets/UTM/utm1700b.mtx'] # 1700*1700
    # dataSetNames=['UTM']
    # srcAVec=['datasets/ECO/wm2.mtx',"datasets/DWAVE/dwa512.mtx","datasets/AST/mcfe.mtx",'datasets/UTM/utm1700a.mtx','datasets/RDB/rdb2048.mtx','datasets/ZENIOS/zenios.mtx','datasets/QCD/qcda_small.mtx',"datasets/BUS/gemat1.mtx",]
    # srcBVec=['datasets/ECO/wm3.mtx',"datasets/DWAVE/dwb512.mtx","datasets/AST/mcfe.mtx",'datasets/UTM/utm1700b.mtx','datasets/RDB/rdb2048l.mtx','datasets/ZENIOS/zenios.mtx','datasets/QCD/qcdb_small.mtx',"datasets/BUS/gemat1.mtx",]
    # dataSetNames=['ECO','DWAVE','AST','UTM','RDB','ZENIOS','QCD','BUS']
    # srcAVec=['datasets/ECO/wm2.mtx',"datasets/DWAVE/dwa512.mtx","datasets/AST/mcfe.mtx",'datasets/UTM/utm1700a.mtx','datasets/RDB/rdb2048.mtx','datasets/ZENIOS/zenios.mtx','datasets/QCD/qcda_small.mtx',"datasets/BUS/gemat1.mtx",]
    # srcBVec=['datasets/ECO/wm3.mtx',"datasets/DWAVE/dwb512.mtx","datasets/AST/mcfe.mtx",'datasets/UTM/utm1700b.mtx','datasets/RDB/rdb2048l.mtx','datasets/ZENIOS/zenios.mtx','datasets/QCD/qcdb_small.mtx',"datasets/BUS/gemat1.mtx",]
    # aRowVec= [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    aRowVec = [50, 100, 500, 1000, 2000, 4000, 5000]
    # aRowVec=[100, 200, 500, 1000]
    # add the algo tag here
    algosVec = ['flat', 'LSH-H', 'Flann', 'PQ', 'IVFPQ', 'onlinePQ', 'HNSW', 'NSW', 'NSG', 'nnDescent', 'DPG']
    # algosVec = ['flat', 'LSH-H']
    # algosVec = ['flat', 'onlinePQ']
    # algosVec=['incrementalRaw']
    # algosVec=[ 'pq']
    # algoDisp = ['BrutalForce', 'PQ']
    algoDisp = ['Baseline', 'LSH', 'Flann', 'PQ', 'IVFPQ', 'onlinePQ', 'HNSW', 'NSW', 'NSG', 'nnDescent', 'DPG']
    # algoDisp = ['BrutalForce', 'LSH-H']
    # algoDisp=['BrutalForce']
    # algoDisp=['PQ']
    # add the algo tag here

    # this template configs all algos as lazy mode, all datasets are static and normalized
    csvTemplate = 'config_e2e_static_lazy.csv'
    # do not change the following
    resultPaths = algosVec
    os.system("mkdir ../../results")
    os.system("mkdir ../../figures")
    os.system("mkdir " + figPath)
    # run
    reRun = 0
    if (len(sys.argv) < 2):

        os.system("sudo rm -rf " + commonBasePath)

        reRun = 1
    else:
        reRun = int(sys.argv[1])
    os.system("sudo mkdir " + commonBasePath)
    print(reRun)
    methodTags = algoDisp
    elapsedTimeAll, incrementalBuildAll, periodAll, recall, incrementalSearchAll, pendingWaitTimeAll, l2StallAll, l3StallAll, totalStallAll, froAll = compareMethod(
        exeSpace, commonBasePath, resultPaths, csvTemplate, algosVec, aRowVec, reRun)
    # Add some pre-process logic for int8 here if it is used

    # groupBar2.DrawFigureYLog(aRowVec, recall/recall[-1], methodTags, "Datasets", "Ins (times of LTMM)", 5, 15, figPath + "/" + "recall", True)
    # groupBar2.DrawFigureYLog(aRowVec, fpInsAll/fpInsAll[-1], methodTags, "Datasets", "FP Ins (times of LTMM)", 5, 15, figPath + "/" + "FP_recall", True)
    # groupBar2.DrawFigureYLog(aRowVec, memInsAll/memInsAll[-1], methodTags, "Datasets", "Mem Ins (times of LTMM)", 5, 15, figPath + "/" + "mem_recall", True)
    # groupBar2.DrawFigure(aRowVec, ratioFpIns, methodTags, "Datasets", "SIMD Utilization (%)", 5, 15, figPath + "/" + "SIMD utilization", True)
    # groupBar2.DrawFigure(aRowVec, recall/(memLoadAll+memStoreAll), methodTags, "Datasets", "IPM", 5, 15, figPath + "/" + "IPM", True)
    # groupBar2.DrawFigure(aRowVec, fpInsAll/(memLoadAll+memStoreAll), methodTags, "Datasets", "FP Ins per Unit Mem Access", 5, 15, figPath + "/" + "FPIPM", True)
    # groupBar2.DrawFigure(aRowVec, (memLoadAll+memStoreAll)/(recall)*100.0, methodTags, "Datasets", "Ratio of Mem Ins (%)", 5, 15, figPath + "/" + "mem", True)

    # groupBar2.DrawFigure(aRowVec, branchAll/recall*100.0, methodTags, "Datasets", "Ratio of Branch Ins (%)", 5, 15, figPath + "/" + "branches", True)
    # groupBar2.DrawFigure(aRowVec, otherIns/recall*100.0, methodTags, "Datasets", "Ratio of Other Ins (%)", 5, 15, figPath + "/" + "others", True)
    # print(recall[-1],recall[2])

    # groupBar2.DrawFigure(dataSetNames, np.log(thrAll), methodTags, "Datasets", "elements/ms", 5, 15, figPath + "sec4_1_e2e_static_lazy_throughput_log", True)
    groupLine.DrawFigureYLog(periodAll, incrementalBuildAll / 1000,
                             methodTags,
                             "EventRate", r'95% Latency of insert (ms)', 0, 1,
                             figPath + "/" + "scanIPEventRate_lat_INSERT_feedMode",
                             True)
    groupLine.DrawFigureYLog(periodAll, pendingWaitTimeAll / 1000,
                             methodTags,
                             "EventRate", r'Pending wait for insert (ms)', 0, 1,
                             figPath + "/" + "scanIPEventRate_lat_pending_feedMode",
                             True)
    groupLine.DrawFigureYLog(periodAll, incrementalSearchAll / 1000,
                             methodTags,
                             "EventRate", r'Latency of search (ms)', 0, 1,
                             figPath + "/" + "scanIPEventRate_lat_search_feedMode",
                             True)
    groupLine.DrawFigureYLog(periodAll, (incrementalSearchAll + pendingWaitTimeAll) / 1000,
                             methodTags,
                             "EventRate", r'Latency of query (ms)', 0, 1,
                             figPath + "/" + "scanIPEventRate_lat_instant_search",
                             True)
    groupLine.DrawFigureYnormal(periodAll, recall,
                                methodTags,
                                "EventRate", r'recall@10', 0, 1,
                                figPath + "/" + "scanIPEventRate_recall_feedMode",
                                True)


if __name__ == "__main__":
    main()
