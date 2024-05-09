import torch
import PyCANDY as candy
import os
import sys
import signal
import sys
import time
cpath=""
def timeout_handler(signum, frame):
    global cpath
    print("Time's up! Exiting now.")
    a=int(0)
    dict={'cpuCycle':a,'cpuCycle':a,'l1dStall':a,'l2Stall':a,'l3Stall':a,'memStall':a,'perfElapsedTime':a,'totalStall':a}
    cfg= candy.dictToConfigMap(dict)
    cfg.toFile('perfInsert.csv')
    cfg.toFile('perfQuery.csv')
    os.system("sudo cp *.csv "+cpath)
    sys.exit(0)  # Exit the program

def set_wall_timer(hours):
    seconds = hours * 3600
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)  # Set the alarm
def runAKNN(cfgName,copyPath):
    cfg=candy.ConfigMap()
    cfg.fromFile(cfgName)
    cfgd=candy.configMapToDict(cfg)
    aknn = candy.createIndex(cfgd['indexTag'])
    dl = candy.createDataLoader(cfgd['dataLoaderTag'])
    perf0 = candy.PAPIPerf()
    perf0.initEventsByCfg(cfg)
    perf1 = candy.PAPIPerf()
    perf1.initEventsByCfg(cfg)
    aknn.setConfig(cfg)
    dl.setConfig(cfg)
    data=dl.getData()
    query=dl.getQuery()
    ini=data[0:5000,:]
    rst=data[5000:,:]
    aknn.loadInitialTensor(ini)
    # perf the insert
    perf0.start()
    aknn.insertTensor(rst)
    perf0.end()
    rucsv=perf0.resultToConfigMap()
    rucsv.toFile('perfInsert.csv')
    #perf the query
    perf1.start()
    aknn.searchTensor(query,cfgd['ANNK'])
    perf1.end()
    rucsv=perf1.resultToConfigMap()
    rucsv.toFile('perfQuery.csv')
    os.system("sudo cp *.csv "+copyPath)
def main():
    global cpath
    cpath=sys.argv[2]
    set_wall_timer(4)
    runAKNN(sys.argv[1],sys.argv[2])
if __name__ == "__main__":
    main()