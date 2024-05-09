#!/usr/bin/env python3
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
from runAKNN import *
OPT_FONT_NAME = 'Helvetica'
TICK_FONT_SIZE = 22
LABEL_FONT_SIZE = 22
LEGEND_FONT_SIZE = 22
LABEL_V = FontProperties(style='normal', size=LABEL_FONT_SIZE)
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
    os.system('rm *.csv')
    os.system('cp config_template.csvt config_template.csv')
    configTemplate = "config_template.csv"
    # clear old files
    
    # editConfig(configTemplate,  configFname, "earlierEmitMs", 0)
    editConfig(configTemplate, "temp2.csv", "vecVolume", prefixTag)
    editConfig("temp2.csv", "temp1.csv", "indexTag", algoTag)
    if (algoTag == 'LSH'):
        editConfig( "temp1.csv",  "temp2.csv", "numberOfBuckets", 1)
        editConfig( "temp2.csv",  "temp3.csv", "useCRS", 0)
        editConfig( "temp3.csv",  "temp4.csv", "congestionDropWorker_algoTag", "onlineIVFLSH")
        editConfig( "temp4.csv",  "temp1.csv", "encodeLen", 3)
    if (algoTag == 'LSH-H'):
        editConfig( "temp1.csv",  "temp2.csv", "indexTag", "onlineIVFLSH")
        editConfig( "temp2.csv",  "temp4.csv", "useCRS", 0)
        editConfig( "temp4.csv",  "temp1.csv", "encodeLen", 3)
    if (algoTag == 'flatAMMIP'):
        editConfig( "temp1.csv",  "temp2.csv", "congestionDropWorker_algoTag", "flatAMMIP")
        editConfig( "temp2.csv",  "temp1.csv", "sketchSize", 256)
    if (algoTag == 'flatAMMIPSMPPCA'):
        editConfig( "temp1.csv",  "temp2.csv", "congestionDropWorker_algoTag", "flatAMMIP")
        editConfig( "temp2.csv",  "temp4.csv", "sketchSize", 128)
        editConfig( "temp4.csv",  "temp1.csv", "ammAlgo", 'smp-pca')
    if (algoTag == 'flat'):
        editConfig( "temp1.csv",  "temp2.csv", "congestionDropWorker_algoTag", "flat")
        editConfig( "temp2.csv",  "temp1.csv", "sketchSize", 256)
    if (algoTag == 'NSW'):
        editConfig( "temp1.csv",  "temp2.csv", "congestionDropWorker_algoTag", "NSW")
        editConfig( "temp2.csv",  "temp1.csv", "is_NSW", 1)
    if (algoTag == 'nnDescent'):
        editConfig( "temp1.csv",  "temp2.csv", "congestionDropWorker_algoTag", "nnDescent")
        editConfig( "temp2.csv",  "temp1.csv", "frozenLevel", 1)
    if (algoTag == 'onlinePQ'):
        editConfig( "temp1.csv",  "temp3.csv", "faissIndexTag", "PQ")
        editConfig( "temp3.csv",  "temp2.csv", "isOnlinePQ", 1)
        editConfig( "temp2.csv",  "temp1.csv", "sketchSize", 256)
    if (algoTag == 'Flann'):
        editConfig( "temp1.csv",  "temp2.csv", "congestionDropWorker_algoTag", "Flann")
        editConfig( "temp2.csv",  "temp1.csv", "sketchSize", 256)
    if (algoTag == 'DPG'):
        editConfig( "temp1.csv",  "temp2.csv", "congestionDropWorker_algoTag", "DPG")
        editConfig( "temp2.csv",  "temp1.csv", "frozenLevel", 1)
    
    # prepare new file
    os.system("sudo mkdir " + resultPath + "/" + str(prefixTag))
    # run
    #runAKNN("temp1.csv",resultPath + "/" + str(prefixTag))
    os.system("export OMP_NUM_THREADS=1 &&" + "python3 runAKNN.py temp1.csv "+resultPath + "/" + str(prefixTag))
    # copy result
    os.system('rm *.csv')


def runPeriodVector(exePath, algoTag, resultPath, prefixTag, configTemplate="config_template.csv", reRun=1):
    for i in range(len(prefixTag)):
        if reRun == 2:
            if checkResultSingle(prefixTag[i], resultPath) == 1:
                print("skip " + str(prefixTag[i]))
            else:
                runPeriod(exePath, algoTag, resultPath, configTemplate, prefixTag[i])
        else:
            runPeriod(exePath, algoTag, resultPath, configTemplate, prefixTag[i])




def readResultSingle(singleValue, resultPath):
    resultFname = resultPath + "/" + str(singleValue) + "/perfInsert.csv"
    elapsedTime = readConfig(resultFname, "perfElapsedTime")
    cpuCycle = readConfig(resultFname, "cpuCycle")
    memStall = readConfig(resultFname, "memStall")
    instructions = readConfig(resultFname, "instructions")
    l1dStall = readConfig(resultFname, "l1dStall")
    l2Stall = readConfig(resultFname, "l2Stall")
    l3Stall = readConfig(resultFname, "l3Stall")
    totalStall=readConfig(resultFname, "totalStall")
    return elapsedTime, cpuCycle, memStall, instructions, l1dStall, l2Stall, l3Stall,totalStall
def checkResultSingle(singleValue, resultPath):
    resultFname = resultPath + "/" + str(singleValue) + "/perfInsert.csv"
    ruExists=0
    if os.path.exists(resultFname):
        ruExists=1
    else:
        print("File does not exist:"+resultFname)
        ruExists=0
    return ruExists

def readResultVector(singleValueVec, resultPath):
    elapseTimeVec = []
    cpuCycleVec = []
    memStallVec = []
    instructionsVec = []
    l1dStallVec = []
    l2StallVec = []
    l3StallVec = []
    totalStallVec=[]
    for i in singleValueVec:
        elapsedTime, cpuCycle, memStall, instructions, l1dStall, l2Stall, l3Stall,totalStall = readResultSingle(i, resultPath)
        elapseTimeVec.append(float(elapsedTime) / 1000.0)
        cpuCycleVec.append(float(cpuCycle))
        memStallVec.append(float(memStall))
        instructionsVec.append(float(instructions))
        l1dStallVec.append(float(l1dStall))
        l2StallVec.append(float(l2Stall))
        l3StallVec.append(float(l3Stall))
        totalStallVec.append(float(totalStall))
    return np.array(elapseTimeVec), np.array(cpuCycleVec), np.array(memStallVec), np.array(instructionsVec), np.array(
        l1dStallVec), np.array(l2StallVec), np.array(l3StallVec),np.array(totalStallVec)

def checkResultVector(singleValueVec, resultPath):
    resultIsComplete=0
    for i in singleValueVec:
        resultIsComplete= checkResultSingle(i, resultPath)
        if resultIsComplete==0:
            return 0
    return 1


def compareMethod(exeSpace, commonPathBase, resultPaths, csvTemplate, algos, dataSetName, reRun=1):
    elapsedTimeAll = []
    memStallAll = []
    incrementalSearchAll = []
    periodAll = []
    instructionAll = []
    l1dStallAll = []
    l2StallAll = []
    l3StallAll = []
    totalStallAll = []
    cyclesAll = []
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
            elapsedTime, cpuCycle, memStall, instructions, l1dStall, l2Stall, l3Stall,totalStall = readResultVector(
                dataSetName, resultPath)
            elapsedTimeAll.append(elapsedTime)
            instructionAll.append(instructions)
            memStallAll.append(memStall)
            l1dStallAll.append(l1dStall)
            periodAll.append(dataSetName)
            cyclesAll.append(cpuCycle)
            l2StallAll.append(l2Stall)
            l3StallAll.append(l3Stall)
            totalStallAll.append(totalStall)
            algoCnt = algoCnt + 1
            print(algoCnt)
        # periodAll.append(periodVec)
    return np.array(elapsedTimeAll), np.array(cyclesAll), np.array(periodAll), np.array(instructionAll), np.array(
        memStallAll), np.array(l1dStallAll), np.array(l2StallAll), np.array(l3StallAll), np.array(
        totalStallAll)
def getCyclesPerMethod(cyclesAll, valueChose):
    instructionsPerMethod = []
    for i in range(len(cyclesAll)):
        instructionsPerMethod.append(cyclesAll[int(i)][int(valueChose)])
    return np.array(instructionsPerMethod)       

def main():
    exeSpace = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/"
    commonBasePath = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/results/cycles_breakdown_1/"

    figPath = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/figures/cycles_breakdown_1/"
    
    # add the datasets here
    # srcAVec=["datasets/AST/mcfe.mtx"] # 765*756
    # srcBVec=["datasets/AST/mcfe.mtx"] # 765*756
    # dataSetNames=['AST']
    #srcAVec=['datasets/UTM/utm1700a.mtx'] # 1700*1700
    #srcBVec=['datasets/UTM/utm1700b.mtx'] # 1700*1700
    #dataSetNames=['UTM']
    #algosVec=['crs', 'mm']
    #algoDisp=['CRS','LTMM']
    #algoDisp=['INT8', 'CRS', 'CS', 'CoOFD', 'BlockLRA', 'FastJLT', 'VQ', 'PQ', 'RIP', 'SMP-PCA', 'WeightedCR', 'TugOfWar',  'NLMM', 'LTMM']
    aRowVec = [6000, 7000]
    dataSetNames= aRowVec
    # aRowVec=[100, 200, 500, 1000]
    # add the algo tag here
    #algosVec = ['flat', 'LSH-H', 'PQ', 'IVFPQ', 'onlinePQ', 'HNSW', 'NSW', 'NSG', 'nnDescent']
    algosVec = ['flat', 'LSH-H','HNSWNaive']
    # algosVec = ['flat', 'onlinePQ']
    # algosVec=['incrementalRaw']
    # algosVec=[ 'pq']
    # algoDisp = ['BrutalForce', 'PQ']
    #algoDisp = ['Baseline', 'LSH', 'PQ', 'IVFPQ', 'onlinePQ', 'HNSW', 'NSW', 'NSG', 'nnDescent']
    algoDisp = ['Baseline', 'LSH','HNSW']
    # algoDisp=['BrutalForce']
    # algoDisp=['PQ']
    # add the algo tag here
    # add the algo tag here
    #algosVec=['mm', 'crs', 'countSketch', 'int8', 'weighted-cr', 'rip', 'smp-pca', 'tugOfWar', 'blockLRA', 'vq', 'pq', 'fastjlt', 'cooFD', 'int8_fp32']
    
    # this template configs all algos as lazy mode, all datasets are static and normalized
    csvTemplate = 'config_e2e_static_lazy.csv'
    # do not change the following
    resultPaths = algosVec

    # run
    reRun = 0
    os.system("mkdir ../../results")
    os.system("mkdir ../../figures")
    os.system("mkdir " + figPath)
    if (len(sys.argv) < 2):
        
        os.system("sudo rm -rf " + commonBasePath)
        os.system("sudo mkdir " + commonBasePath)
        reRun = 1
    else:
        reRun=int(sys.argv[1])
    os.system("sudo mkdir " + commonBasePath)
    print(reRun)
    #exit()
    methodTags =algoDisp
    elapsedTimeAll, cpuCycleAll, periodAll, instructions, memStallAll, l1dStallAll, l2StallAll, l3StallAll,totalStallAll = compareMethod(
        exeSpace, commonBasePath, resultPaths, csvTemplate, algosVec, aRowVec, reRun)
    # Add some pre-process logic for int8 here if it is used

    #print(instructions)
    print(memStallAll)
    #exit(0)
    # adjust int8: int8/int8_fp32*mm
    
   
    otherStallsAll = totalStallAll-memStallAll-l1dStallAll-l2StallAll-l3StallAll
    otherStallsAll = np.maximum(otherStallsAll,0)
    print(otherStallsAll)
    totalStallAll = memStallAll+l1dStallAll+l2StallAll+l3StallAll+otherStallsAll
    nonStallAll=cpuCycleAll-totalStallAll
    nonStallAll=np.maximum(nonStallAll,0)
    cpuCycleAll=totalStallAll+nonStallAll
    allowLegend = True
    valueVec=dataSetNames
    for valueChose in range(len(valueVec)):
        # instructionsPerMethod=getCyclesPerMethod(instructions,valueChose)
        cpuCyclePerMethod = getCyclesPerMethod(cpuCycleAll, valueChose)
        memStallPerMethod = getCyclesPerMethod(memStallAll, valueChose)
        l1dStallPerMethod = getCyclesPerMethod(l1dStallAll, valueChose)
        l2StallPerMethod = getCyclesPerMethod(l2StallAll, valueChose)
        l3StallPerMethod = getCyclesPerMethod(l3StallAll, valueChose)
        totalStallAllPerMethod=getCyclesPerMethod(totalStallAll, valueChose)
        otherPerMethod = getCyclesPerMethod(otherStallsAll, valueChose)
        nonStallPerMethod=getCyclesPerMethod(nonStallAll, valueChose)
        accuBar.DrawFigure(methodTags,
                           [memStallPerMethod, l1dStallPerMethod, l2StallPerMethod, l3StallPerMethod,
                            otherPerMethod,nonStallPerMethod]/cpuCyclePerMethod*100.0, ['Mem Stall', 'L1D Stall', 'L2 Stall', 'L3 Stall', 'Other Stall', 'Not Stall'], '',
                           'Propotion (%)', figPath + "/" + "cyclesbreakDown"
                           + "_cycles_accubar" + str(valueVec[valueChose]), allowLegend,
                           '')
        
        allowLegend = False
        
    #draw2yBar(methodTags,[lat95All[0][0],lat95All[1][0],lat95All[2][0],lat95All[3][0]],[errAll[0][0],errAll[1][0],errAll[2][0],errAll[3][0]],'95% latency (ms)','Error (%)',figPath + "sec6_5_stock_q1_normal")
    #groupBar2.DrawFigure(dataSetNames, errAll, methodTags, "Datasets", "Error (%)", 5, 15, figPath + "sec4_1_e2e_static_lazy_fro", True)
    #groupBar2.DrawFigure(dataSetNames, np.log(lat95All), methodTags, "Datasets", "95% latency (ms)", 5, 15, figPath + "sec4_1_e2e_static_lazy_latency_log", True)
    ipcAll=instructions/cpuCycleAll
    groupBar2.DrawFigure(dataSetNames,ipcAll,methodTags, "Datasets", "IPC", 5, 15, figPath + "ipc", True)
    groupBar2.DrawFigure(dataSetNames,totalStallAll/cpuCycleAll*100.0,methodTags, "Datasets", "Ratio of Stalls (%)", 5, 15, figPath + "stall_ratio", True)
    groupBar2.DrawFigure(dataSetNames,l1dStallAll/cpuCycleAll*100.0,methodTags, "Datasets", "Ratio of l1dStalls (%)", 5, 15, figPath + "l1dstall_ratio", True)
    groupBar2.DrawFigure(dataSetNames,l2StallAll/cpuCycleAll*100.0,methodTags, "Datasets", "Ratio of l2Stalls (%)", 5, 15, figPath + "l2stall_ratio", True)
    groupBar2.DrawFigure(dataSetNames,l3StallAll/cpuCycleAll*100.0,methodTags, "Datasets", "Ratio of l3Stalls (%)", 5, 15, figPath + "l3stall_ratio", True)
    groupBar2.DrawFigureYLog(dataSetNames,memStallAll,methodTags, "Datasets", "Count of memory stall", 5, 15, figPath + "cnt_mem_stall", True)
    print(ipcAll)
    #groupBar2.DrawFigure(dataSetNames,(l1dStallAll+l2StallAll+l3StallAll)/cpuCycleAll*100.0,methodTags, "Datasets", "Ratio of cacheStalls (%)", 5, 15, figPath + "cachestall_ratio", True)



    
    #groupBar2.DrawFigure(dataSetNames, np.log(thrAll), methodTags, "Datasets", "elements/ms", 5, 15, figPath + "sec4_1_e2e_static_lazy_throughput_log", True)
if __name__ == "__main__":
    main()
