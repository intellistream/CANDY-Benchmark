import torch
import PyCANDY as candy
import os
import sys
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
    # perf the insert
    perf0.start()
    aknn.insertTensor(data)
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
    runAKNN(sys.argv[1],sys.argv[2])
if __name__ == "__main__":
    main()