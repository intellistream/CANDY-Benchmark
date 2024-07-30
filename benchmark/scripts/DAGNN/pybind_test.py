import PyCANDY
import torch
import time
import intelli_timestamp_generator
import torch_helper
from intelli_timestamp_generator import *
from torch_helper import *
def encode_states_to_tensor(state:PyCANDY.GraphStates, vecDim):
    gs = state.global_stat
    bs = state.time_local_stat
    ws = state.window_states

    combined_gs = [gs.degree_sum/gs.ntotal, gs.degree_variance, gs.neighbor_distance_sum/gs.ntotal, gs.neighbor_distance_variance, gs.ntotal, gs.steps_expansion_average, gs.steps_taken_avg, gs.steps_taken_max]
    combined_bs = [bs.degree_sum_new/bs.ntotal, bs.degree_sum_old/bs.old_ntotal, bs.degree_variance_new, bs.degree_variance_old, bs.neighbor_distance_sum_new/bs.ntotal, bs.neighbor_distance_sum_old/bs.old_ntotal, bs.neighbor_distance_variance_new, bs.neighbor_distance_variance_old, bs.ntotal, bs.steps_expansion_sum/bs.ntotal, bs.steps_taken_max, bs.steps_taken_sum/bs.ntotal]
    combined_ws = [ws.hierarchyWindowSize, ws.newWindowSize, ws.oldWindowSize, ws.getCount(0), ws.getCount(1), ws.getCount(2)]

    combined_state = combined_gs+combined_bs+combined_ws
    state_tensor = torch.tensor(combined_state)
    return state_tensor

cfg = PyCANDY.ConfigMap()
cfg.fromFile("./config.csv")
cfgd = PyCANDY.configMapToDict(cfg)
vecDim = cfgd['vecDim']
ANNK = cfgd['ANNK']


dagnn = PyCANDY.DAGNNIndex()
dagnn.setConfig(cfg)

dl = PyCANDY.createDataLoader(cfgd['dataLoaderTag'])
initialRows = cfgd['initialRows']
dl.setConfig(cfg)

dataTensorAll = dl.getData().nan_to_num(0)
dataTensorInitial = dataTensorAll[:initialRows]
dataTensorStream = dataTensorAll[initialRows:]
queryTensor = dl.getQuery()




timestampGen = intelli_timestamp_generator.IntelliTimeStampGenerator(dataTensorStream.size(0))
timestamps = timestampGen.getTimeStamps()
print(f"timestamp size =" + str(len(timestamps)))

batchSize = cfgd['batchSize']

startRow = 0
endRow = startRow + batchSize
tNow = 0
tExpectedArrival = timestamps[endRow-1].arrivalTime
tp = 0
tDone = 0
aRows = dataTensorStream.size(0)

print("3.0 Load initial Tensor!!!")

if(initialRows > 0 ):
    dagnn.loadInitialTensor(dataTensorInitial)
print("3.1 Stream now!!!")
start = time.perf_counter()
processOld = 0
while startRow<aRows:
    tNow = (time.perf_counter() - start)*1e6
    while(tNow<tExpectedArrival):
        tNow = (time.perf_counter() - start)*1e6
    processed = endRow
    processed = processed*100.0/aRows
    subA = dataTensorStream[startRow:endRow]
    dagnn.insertTensor(subA)
    state = dagnn.getState()
    print(encode_states_to_tensor(state, vecDim))
    tp = time.perf_counter()-start
    for i in range(startRow, endRow, 1):
        timestamps[i].processedTime = tp
    startRow += batchSize
    endRow += batchSize

    if(endRow>aRows):
        endRow = aRows
    if(processed-processOld >= 10.0):
        print("Done" + str(processed)+"%("+str(startRow)+"/"+str(aRows)+")")
        processOld = processed
    tExpectedArrival = timestamps[endRow-1].arrivalTime

print("Insert is done, let us validate the results")
startQuery = time.perf_counter()
indexResults = dagnn.searchTensor(queryTensor, ANNK)
tNow = (time.perf_counter()-startQuery)*1e6
print("Query done in "+str(tNow) + "ms")
queryLatency = tNow

gdIndex = PyCANDY.createIndex('flat')
gdIndex.setConfig(cfg)
if(initialRows>0):
    gdIndex.loadInitialTensor(dataTensorInitial)
gdIndex.insertTensor(dataTensorStream)

gdResults = gdIndex.searchTensor(queryTensor, ANNK)




recall = torch_helper.calculate_recall(gdResults, indexResults)
print("RECALL = "+str(recall))




