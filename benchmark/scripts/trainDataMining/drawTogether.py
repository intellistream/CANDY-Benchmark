#!/usr/bin/env python3
# Note: the concept drift is not learnt by indexing in this group
import csv
import numpy as np
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



def runPeriod(exePath, algoTag, resultPath, configTemplate="config.csv", prefixTagRaw="null"):
    # resultFolder="periodTests"
    prefixTag = str(prefixTagRaw)
    configFname = "config_period_" + str(prefixTag) + ".csv"
    configTemplate = "config_e2e_static_lazy.csv"
    # clear old files
    os.system("cd " + exePath + "&& sudo rm *.csv")
    os.system("cp perfListEvaluation.csv " + exePath)
    # editConfig(configTemplate, exePath + configFname, "earlierEmitMs", 0)
    editConfig(configTemplate, "temp2.csv", algoTag, prefixTag)
    editConfig("temp2.csv", exePath + "temp1.csv", "indexTag", "DAGNN")
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
        scanVec = dataSetName[i]
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
                dataSetName[i], resultPath)
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
    commonBasePath = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/results/trainDataMining/"

    figPath = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/figures/trainDataMining"

    fields = {
        "efConstruction":[20,40,60,80,100,120],
        "efSearch":[8,16,24,32,48,64],
        "rng_alpha":[0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95,1.0],
        "clusterExpansionStep":[1,2,3],
        "optimisticN":[0,8,16,32,48,64,96],
        "discardN":[0,1,2,4,6,8],
        "discardClusterN":[16,32,64,96,144,192,256],
        "discardClusterProp":[0.1,0.2,0.35,0.5,0.75],
        "degree_std_range":[0.5,0.75,1.0,1.25,1.5,1.75,2.25],
        "degree_allow_range":[0.125,0.25,0.375,0.5,0.75],
        "degree_lift_range":[1,1.25,1.5,1.75,2.5],
        "expiration_timestamp":[300,400,500,600,800,1000,1200,1600,2000,2500],
        "max_backtrack_steps":[8,12,16,20,28,36],
        "steps_above_avg":[20,35,50,75,100,125,150],
        "steps_above_max":[10,20,35,50,75,100],
        "nb_navigation_paths":[8,16,24,32,48,64,96]
    }

    algosVec =  fields.keys()
    algoDisp = fields.keys()
    # algoDisp = ['BrutalForce', 'LSH-H']
    # algoDisp=['BrutalForce']
    # algoDisp=['PQ']
    # add the algo tag here

    # this template configs all algos as lazy mode, all datasets are static and normalized
    csvTemplate = 'config_e2e_static_lazy.csv'
    # do not change the following
    resultPaths = list(algosVec)
    os.system("mkdir ../../results")
    os.system("mkdir ../../figures")
    os.system("mkdir " + figPath)
    # run
    aRowVec = list(fields.values())
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
        exeSpace, commonBasePath, resultPaths, csvTemplate, list(algosVec), aRowVec, reRun)
    # Add some pre-process logic for int8 here if it is used


if __name__ == "__main__":
    main()
