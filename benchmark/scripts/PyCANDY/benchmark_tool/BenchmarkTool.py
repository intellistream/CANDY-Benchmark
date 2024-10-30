import PyCANDY as candy
import torch
import time
class BenchmarkTool:
    def __init__(self):
        self.configMap = None  # 存储配置的字典形式
        self.configMapRaw = None
    def loadConfigFromFile(self, fileName,isRef=0,refTag ='flat'):
        """
        从配置文件中读取内容并将其加载为配置字典。
        
        :param fileName: str - 配置文件路径
        :param isRef: int - 是否用作参照而非评估
        :para, refTag: str - 如果用作参照，启动哪一个算法作为基线
        :return: dict - 配置内容的字典表示
        """
        # 创建一个配置映射对象并从文件加载配置
        configMap = candy.ConfigMap()
        configMap.fromFile(fileName)
        self.configMapRaw = configMap
        # 将配置映射对象转换为字典
        self.configMap = candy.configMapToDict(configMap)
        self.indexPtr = None
        # 打印或返回配置字典以便进一步处理
        print("Loaded configuration:", self.configMap)
        dataLoaderTag = self.configMap.get("dataLoaderTag", "random")
        indexTag = self.configMap.get("indexTag", 'flat')
        print(f"Data loader initialized with tag: {dataLoaderTag}")
        self.dataLoader = candy.createDataLoader(dataLoaderTag)
        self.isRef = isRef
        if(isRef==1):
            self.indexPtr =  candy.createIndex(refTag)
            print("This is just a ref")
        else:
            self.indexPtr =  candy.createIndex(indexTag)
        self.indexPtr.setConfig(configMap)
        self.dataLoader.setConfig(configMap)
         # 配置数据加载器
        return self.configMap
    def refFlag(self):
        return self.isRef
    def loadConfigFromDict(self, dict,isRef=0,refTag ='flat'):
        """
        从Python字典读取内容并将其加载为配置字典。
        
        :param dict: Dic - python字典
        :param isRef: int - 是否用作参照而非评估
        :para, refTag: str - 如果用作参照，启动哪一个算法作为基线
        :return: dict - 配置内容的字典表示
        """
        # 创建一个配置映射对象并从文件加载配置
        configMap = candy.dictToConfigMap(dict)
        self.configMapRaw = configMap
        # 将配置映射对象转换为字典
        self.configMap = candy.configMapToDict(configMap)
        self.indexPtr = None
        # 打印或返回配置字典以便进一步处理
        print("Loaded configuration:", self.configMap)
        dataLoaderTag = self.configMap.get("dataLoaderTag", "random")
        indexTag = self.configMap.get("indexTag", 'flat')
        print(f"Data loader initialized with tag: {dataLoaderTag}")
        self.dataLoader = candy.createDataLoader(dataLoaderTag)
        self.isRef = isRef
        if(isRef==1):
            self.indexPtr =  candy.createIndex(refTag)
            print("This is just a ref")
        else:
            self.indexPtr =  candy.createIndex(indexTag)
        self.indexPtr.setConfig(configMap)
        self.dataLoader.setConfig(configMap)
         # 配置数据加载器
        return self.configMap
    def getQueryAndDataTensors(self):
        """
        根据配置字典分离并返回 queryTensor 和 dataTensor。
        
        :return: tuple - (queryTensor, dataTensor)
        """
        # 从配置中获取必要的参数
        
        cutOffTimeSeconds = int(self.configMap.get("cutOffTimeSeconds", -1))
        waitPendingWrite = self.configMap.get("waitPendingWrite", 0)
        vecVolume = int(self.configMap.get("vecVolume", 1000))
        # 获取数据行参数
        initialRows = int(self.configMap.get("initialRows",1))
        print(initialRows)
        print(vecVolume)
        # 获取数据加载器
        
        if self.dataLoader is None:
            print("Error: Data loader could not be found.")
            return None, None
        
       
        
        
        deleteRows = self.configMap.get("deleteRows", 0)
        
        # 获取完整的数据张量并处理 NaN 值
        dataTensorAll = self.dataLoader.getData().nan_to_num(0)
        
        # 分割数据张量
        dataTensorInitial = dataTensorAll[:initialRows].nan_to_num(0)
        dataTensorStream = dataTensorAll[initialRows:].nan_to_num(0)
        
        # 获取查询张量并处理 NaN 值
        queryTensor = self.dataLoader.getQuery().nan_to_num(0)
        #print(dataTensorInitial)
        # 返回 queryTensor 和 dataTensor
        return queryTensor, dataTensorInitial, dataTensorStream
    def generateTimestamps(self, rows, eventRate):
        """
        根据给定的 eventRate 为 tensor 的每行生成均匀递增的事件时间戳和处理时间戳。
        
        :param rows: int - 输入的 2D tensor行数
        :param eventRate: float - 每秒事件数 (行数/秒)
        :return: tuple - (eventTimestamps, processingTimestamps)
        """
        # 计算时间间隔（微秒）
        staticDataSet = int(self.configMap.get("staticDataSet", 0))
        intervalMicros = int(1e6 / eventRate)  # 每行的时间间隔，单位为微秒
        
        # 获取张量的行数
        numRows = rows
        eventTimestamps = None
        if(staticDataSet==1):
              # 生成处理时间戳，初始化为零
            eventTimestamps = torch.zeros(numRows, dtype=torch.int64)
        else:
        # 生成事件时间戳
            eventTimestamps = torch.arange(0, numRows * intervalMicros, intervalMicros, dtype=torch.int64)
        
        return eventTimestamps
    def loadInitial(self):
        initialRows = int(self.configMap.get("initialRows", 0))
        cutOffTimeSeconds = int(self.configMap.get("cutOffTimeSeconds", -1))
          # 开始时间记录
        start_time = time.time()
        
        # 模拟设置超时
        if cutOffTimeSeconds > 0:
            print(f"Allow up to {cutOffTimeSeconds} seconds before termination")
        
        print("Loading initial tensor...")
        
        # 如果有初始行数，则加载初始张量
        if initialRows > 0:
            initialTensor = self.dataLoader.getDataAt(0, initialRows)
            self.indexPtr.loadInitialTensor(initialTensor)  # 加载初始张量到索引
            print("Initial tensor loaded into index.")

        # 计算并记录构建时间
        self.constructionTime = int((time.time() - start_time) * 1e6)  # 转换为微秒
        print(f"Construction time: {self.constructionTime} microseconds")
        # 设置冻结级别
        frozenLevel = self.configMap.get("frozenLevel", 1)
        self.indexPtr.setFrozenLevel(frozenLevel)
        print(f"Frozen level set to: {frozenLevel}")
    def queryProcess(self,q,annk=1):
        """
        查找并返回
        :param q: torch.Tensor -待查找的向量
        :param annk: int - 查找多少个
        :return: List[torch.Tensor]查询结果
        """
        
        ru = self.indexPtr.searchTensor(q,annk)
      
        return ru
    def deleteBatchProcess(self, eventTimestamps, deleteTensor, batchSize,windowObj=None):
        """
        批量删除向量并更新处理时间戳。
        
        :param eventTimestamps: torch.Tensor - 每个向量的事件时间戳
        :param deleteTensor: torch.Tensor - 待删除的向量数据
        :param batchSize: int - 批量大小
        :param windowObj: 窗体对象, 如果非空需要提供setProgressBar函数设置进度条
        :return: torch.Tensor - 更新后的处理时间戳
        """
        # 获取行数（总向量数）
        numRows = deleteTensor.size(0)
        
        # 初始化处理时间戳为零
        processingTimestamps = torch.zeros(numRows, dtype=torch.int64)
        
        # 初始化索引
        startRow = 0
        endRow = batchSize
        prossedOld = 0

        
        start_time = time.time()

        # 批量删除操作
        while startRow < numRows:
            # 获取当前时间，转换为微秒
            tNow = (time.time() - start_time)*1e6
            
            # 等待直到到达批次的预期时间
            tExpectedArrival = eventTimestamps[endRow - 1]
            while tNow < tExpectedArrival:
                tNow = (time.time() - start_time)*1e6

            # 获取当前批次的数据
            subTensor = deleteTensor[startRow:endRow]
            self.indexPtr.deleteTensor(subTensor,1)  # 删除当前批次的张量数据
            
            # 获取当前处理时间戳（微秒）
            tp = (time.time() - start_time)*1e6

            # 更新当前批次的处理时间戳
            processingTimestamps[startRow:endRow] = tp

            # 更新索引
            startRow += batchSize
            endRow += batchSize
            if endRow >= numRows:
                endRow = numRows  # 确保不超出范围
            
            # 计算并显示进度
            processedPercent = endRow * 100.0 / numRows
            if processedPercent - prossedOld >= 10.0:
                print(f"Done {processedPercent:.1f}% ({startRow}/{numRows})")
                prossedOld = processedPercent
                if(windowObj!=None):
                    windowObj.setProgressBar(processedPercent)
            # 更新下一个批次的预期时间
            tExpectedArrival = eventTimestamps[min(endRow - 1, numRows - 1)]
        if(windowObj!=None):
                    windowObj.setProgressBar(100)
        return processingTimestamps
    def insertBatchProcess(self, eventTimestamps, insertTensor, batchSize,windowObj=None):
        """
        批量插入向量并更新处理时间戳。
        
        :param eventTimestamps: torch.Tensor - 每个向量的事件时间戳
        :param insertTensor: torch.Tensor - 插入的向量数据
        :param batchSize: int - 批量大小
        :param windowObj: 窗体对象, 如果非空需要提供setProgressBar函数设置进度条
        :return: torch.Tensor - 更新后的处理时间戳
        """
        # 获取行数（总向量数）
        numRows = insertTensor.size(0)
        
        # 初始化处理时间戳为零
        processingTimestamps = torch.zeros(numRows, dtype=torch.int64)
        
        # 初始化索引
        startRow = 0
        endRow = batchSize
        prossedOld = 0

        # 开始时间记录 
        start_time = time.time()

        # 批量删除操作
        while startRow < numRows:
            # 获取当前时间，转换为微秒
            tNow = (time.time() - start_time)*1e6
            
            # 等待直到到达批次的预期时间
            tExpectedArrival = eventTimestamps[endRow - 1]
            while tNow < tExpectedArrival:
                tNow = (time.time() - start_time)*1e6

            # 获取当前批次的数据
            subTensor = insertTensor[startRow:endRow]
            self.indexPtr.insertTensor(subTensor)  # 当前批次的张量数据
            
            # 获取当前处理时间戳（微秒）
            tp = (time.time() - start_time)*1e6

            # 更新当前批次的处理时间戳
            processingTimestamps[startRow:endRow] = tp

            # 更新索引
            startRow += batchSize
            endRow += batchSize
            if endRow >= numRows:
                endRow = numRows  # 确保不超出范围
            
            # 计算并显示进度
            processedPercent = endRow * 100.0 / numRows
            if processedPercent - prossedOld >= 10.0:
                print(f"Done {processedPercent:.1f}% ({startRow}/{numRows})")
                prossedOld = processedPercent
                if(windowObj!=None):
                    windowObj.setProgressBar(processedPercent)
            # 更新下一个批次的预期时间
            tExpectedArrival = eventTimestamps[min(endRow - 1, numRows - 1)]
        if(windowObj!=None):
                    windowObj.setProgressBar(100)
        return processingTimestamps
from typing import List
import torch

def exist_row(ground_truth_tensor: torch.Tensor, row_tensor: torch.Tensor) -> bool:
    """
    Check if a given row exists in the ground truth tensor.
    """
    return (ground_truth_tensor == row_tensor).all(dim=1).any().item()

def calculateRecall(ground_truth: List[torch.Tensor], prob: List[torch.Tensor]) -> float:
    """
    Calculate recall based on the ground truth and predicted tensors.
    
    :param ground_truth: List[torch.Tensor] - List of tensors representing ground truth
    :param prob: List[torch.Tensor] - List of tensors representing predicted results
    :return: float - Calculated recall value
    """
    true_positives = 0
    false_negatives = 0

    # Iterate over each tensor in prob
    for i in range(len(prob)):
        gd_i = ground_truth[i]
        prob_i = prob[i]

        # Check each row in prob_i to see if it exists in gd_i
        for j in range(prob_i.size(0)):
            if exist_row(gd_i, prob_i[j]):
                true_positives += 1
            else:
                false_negatives += 1

    # Calculate recall
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    return recall
def getLatencyPercentile(fraction: float, event_time: torch.Tensor, processed_time: torch.Tensor) -> int:
    """
    Calculate the latency percentile from event and processed time tensors.
    
    :param fraction: float - Percentile in the range 0 ~ 1
    :param event_time: torch.Tensor - int64 tensor of event arrival timestamps
    :param processed_time: torch.Tensor - int64 tensor of processed timestamps
    :return: int - The latency value at the specified percentile
    """
    # Calculate latency for valid entries where processed_time >= event_time and processed_time != 0
    valid_latency = (processed_time - event_time)[(processed_time >= event_time) & (processed_time != 0)]

    # If no valid latency, return 0 as in the C++ code
    if valid_latency.numel() == 0:
        print("Error: No valid latency found.")
        return 0

    # Sort the valid latency values
    valid_latency_sorted = torch.sort(valid_latency).values

    # Calculate the index for the percentile
    t = len(valid_latency_sorted) * fraction
    idx = int(t) if int(t) < len(valid_latency_sorted) else len(valid_latency_sorted) - 1

    # Return the latency at the desired percentile
    return valid_latency_sorted[idx].item()