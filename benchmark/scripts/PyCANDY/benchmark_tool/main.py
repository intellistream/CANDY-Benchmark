import sys,time
from BenchmarkTool import BenchmarkTool,calculateRecall,getLatencyPercentile
import PyCANDY as candy
class BenchmarkTask():
    def __init__(self, benchmark_tool, config_map, parent=None,windowObj=None):

        self.benchmark_tool = benchmark_tool
        self.config_map = config_map
        self.windowObj = windowObj
        self.resultTensor= None
    def getQueryResult(self):
        return self.resultTensor
    def run(self):
        # 模拟长时间的基准测试逻辑
        batchSize = int(self.config_map.get("batchSize", 1000))
        eventRate = int(self.config_map.get("eventRateTps", 1000))
        querySize = int(self.config_map.get("querySize", 100))
        self.querySize = querySize
        deleteRows = int(self.config_map.get("deleteRows", 0))
        self.deleteRows = deleteRows
        annk = int(self.config_map.get("ANNK", 10))
        print(f" Batch size = {batchSize},event rata = {eventRate}, annk = {annk},query size ={querySize}")
        print("Benchmark started...")
        queryTensor, dataTensorInitial, dataTensorStream = self.benchmark_tool.getQueryAndDataTensors()
        self.benchmark_tool.loadInitial()
        print("Done loading initial tensor")
        # 删除向量处理
        if deleteRows > 0:
            print("Start delete process")
            deleteTensor = dataTensorInitial[:deleteRows]
            self.eventTimestampsDelete = self.benchmark_tool.generateTimestamps(deleteTensor.size(0), eventRate)
            self.processedTimeStampsDelete=self.benchmark_tool.deleteBatchProcess(self.eventTimestampsDelete,deleteTensor,batchSize )
            

        # 插入处理
        print("Start insert process")
        self.eventTimestampsInsert = self.benchmark_tool.generateTimestamps(dataTensorStream.size(0), eventRate)
        self.processedTimeStampsInsert = self.benchmark_tool.insertBatchProcess(self.eventTimestampsInsert, dataTensorStream, batchSize)
        print("Run query process")
        start_time = time.time()
        self.resultTensor = self.benchmark_tool.queryProcess(queryTensor,annk)
        self.queryTime = int((time.time() - start_time) * 1e6)  # 转换为微秒
        print("Benchmark completed.")
    def getLatency(self,percentile):
        latDelete = 0
        if(self.deleteRows>0):
            latDelete = getLatencyPercentile(percentile,self.eventTimestampsDelete,self.processedTimeStampsDelete)
        latInsert = getLatencyPercentile(percentile,self.eventTimestampsInsert,self.processedTimeStampsInsert)
        return latDelete,latInsert
    def genCommonStatistics(self,recall):
        resultDic = {}
        resultDic ['recall'] = float(recall)
        latDelete,latInsert = self.getLatency(0.95)
        resultDic ['95%latency(Insert)'] = float(latInsert)
        resultDic ['95%latency(Del)'] = float(latDelete)
        resultDic['QPS'] = float(self.querySize *1e6) / (self.queryTime)
        return resultDic,candy.dictToConfigMap(resultDic)
def main():
    file_name = sys.argv[1]
    benchmarkTool = BenchmarkTool() 
    refTool = BenchmarkTool() 
    configMap = benchmarkTool.loadConfigFromFile(file_name)
    configMap2 =  refTool.loadConfigFromFile(file_name,1,'flat')
    annsRun =  BenchmarkTask(benchmarkTool, configMap,windowObj=None)
    refRun = BenchmarkTask(refTool, configMap2,windowObj=None)
    annsRun.run()
    refRun.run()
    annsResult = annsRun.getQueryResult()
    baselineResult = refRun.getQueryResult()
    #print(annsResult[0],baselineResult[0])
    recall = calculateRecall(baselineResult,annsResult)
    dic,cfg=annsRun.genCommonStatistics(recall)
    #print(dic)
    cfg.toFile('result.csv')
    print(f"recall={recall}")
if __name__ == "__main__":
    main()
