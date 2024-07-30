import PyCANDY
import torch
import time
from intelli_timestamp_generator import *
from drl_utils import ReplayBuffer
from DQN import *
from torch_helper import *
def encode_states_to_tensor(state:PyCANDY.GraphStates, vecDim):
    gs = state.global_stat
    bs = state.time_local_stat
    ws = state.window_states

    combined_gs = [gs.degree_sum/(bs.ntotal+bs.old_ntotal), gs.degree_variance, gs.neighbor_distance_sum/(bs.ntotal+bs.old_ntotal), gs.neighbor_distance_variance, gs.ntotal, gs.steps_expansion_average, gs.steps_taken_avg, gs.steps_taken_max]
    combined_bs = [bs.degree_sum_new/bs.ntotal, bs.degree_sum_old/bs.old_ntotal if bs.old_ntotal!=0 else 0, bs.degree_variance_new, bs.degree_variance_old, bs.neighbor_distance_sum_new/bs.ntotal, bs.neighbor_distance_sum_old/bs.old_ntotal if bs.old_ntotal!=0 else 0, bs.neighbor_distance_variance_new, bs.neighbor_distance_variance_old, bs.ntotal, bs.steps_expansion_sum/bs.ntotal, bs.steps_taken_max, bs.steps_taken_sum/bs.ntotal]
    combined_ws = [ws.hierarchyWindowSize, ws.newWindowSize, ws.oldWindowSize, ws.getCount(0), ws.getCount(1), ws.getCount(2)]

    combined_state = combined_gs+combined_bs+combined_ws
    #state_tensor = torch.tensor(combined_state, dtype=torch.float)
    return combined_state

# 0.1 Initialize DQN agent
lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 1000
minimal_size = 100
sample_batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

replay_buffer = ReplayBuffer(buffer_size)
state_dim =26
action_dim = 41
agent = DQN(state_dim, action_dim, hidden_dim, lr, gamma, epsilon, target_update, device)
return_list = []

# 0.2 Intialize AKNN
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

timestampGen = IntelliTimeStampGenerator(dataTensorStream.size(0))
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
Actions = []
num_episodes = (aRows-initialRows)//batchSize
state = encode_states_to_tensor(dagnn.getState(), vecDim)
for i_episode in range(num_episodes):
    tNow = (time.perf_counter() - start)*1e6
    while(tNow<tExpectedArrival):
        tNow = (time.perf_counter() - start)*1e6
    processed = endRow
    processed = processed*100.0/aRows
    episode_return = 0
    action = agent.take_action(state)
    Actions.append(action)
    dagnn.performAction(action)
    subA = dataTensorStream[startRow:endRow]
    dagnn.insertTensor(subA)
    tp=(time.perf_counter()-start)*1e6
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
    # update replay_buffer
    next_state = encode_states_to_tensor(dagnn.getState(), vecDim)
    reward = -tp
    done = False
    replay_buffer.add(state, action, reward, next_state,done)
    state=next_state
    episode_return += reward

    if replay_buffer.size()>minimal_size:
        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(sample_batch_size)
        transition_dict = {
            'states': b_s,
            'actions': b_a,
            'next_states': b_ns,
            'rewards': b_r,
            'dones': b_d
        }
        agent.update(transition_dict)
tDone = time.perf_counter()-start
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


throughput = aRows * 1e6 / tDone
throughputByElements = throughput * dataTensorStream.size(1)
insertion_latency95 = get_latency_percentage(0.95, timestamps)
recall = calculate_recall(gdResults, indexResults)
print("RECALL = "+str(recall))
print("INSERTION LATENCY 95%= "+str(insertion_latency95))
print("QUERY LATENCY = "+str(queryLatency))
print("ACTIONS = " +str(Actions))





