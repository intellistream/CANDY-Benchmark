import PyCANDY
import torch
import time
from intelli_timestamp_generator import *
from morl_baselines.multi_policy.envelope import envelope
import numpy as np
from morl_baselines.common.weights import equally_spaced_weights, random_weights
from morl_baselines.common.evaluation import (
    log_all_multi_policy_metrics,
    log_episode_info,
)
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

state_dim =26
action_dim = 45


#agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
agent = envelope.Envelope(
    max_grad_norm=0.1,
    learning_rate=3e-4,
    gamma=0.98,
    batch_size=64,
    net_arch=[256, 256, 256, 256],
    buffer_size=int(2e3),
    initial_epsilon=1.0,
    final_epsilon=0.05,
    epsilon_decay_steps=50000,
    initial_homotopy_lambda=0.0,
    final_homotopy_lambda=1.0,
    homotopy_decay_steps=10000,
    learning_starts=100,
    envelope=True,
    gradient_updates=1,
    target_net_update_freq=1000,  # 1000,  # 500 reduce by gradient updates
    tau=1,
    log=True,
    project_name="CANDY",
    experiment_name="DAGNN-onlineInsert",
)

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

reset_num_timesteps = True
eval_freq: int = 10000
num_eval_weights_for_front: int = 100
num_eval_episodes_for_front: int = 5
num_eval_weights_for_eval: int = 50
reset_learning_starts: bool = False
verbose: bool = False    

agent.global_step = 0 if reset_num_timesteps else agent.global_step
agent.num_episodes = 0 if reset_num_timesteps else agent.num_episodes
if reset_learning_starts:  # Resets epsilon-greedy exploration
    agent.learning_starts = agent.global_step

num_episodes = 0
eval_weights = equally_spaced_weights(agent.reward_dim, n=num_eval_weights_for_front)
print("3.1 Stream now!!!")
start = time.perf_counter()
processOld = 0
Actions = []
num_episodes = (aRows-initialRows)//batchSize
obs = encode_states_to_tensor(dagnn.getState(), vecDim)
w = random_weights(agent.reward_dim, 1, dist="gaussian", rng=agent.np_random)
tensor_w = torch.tensor(w).float().to(agent.device)
for i_episode in range(num_episodes):
    tNow = (time.perf_counter() - start)*1e6
    while(tNow<tExpectedArrival):
        tNow = (time.perf_counter() - start)*1e6
    processed = endRow
    processed = processed*100.0/aRows
    if agent.global_step < agent.learning_starts:
                action = np.random.randint(agent.action_dim)
    else:
        action = agent.act(obs.float().to(agent.device), tensor_w)
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
    vec_reward = []
    next_obs = encode_states_to_tensor(dagnn.getState(), vecDim)

    agent.global_step+=1
    agent.replay_buffer.add(obs, action, vec_reward, next_obs, False)
    if(agent.global_step>=agent.learning_starts):
         agent.update()
    
    obs = next_obs
    



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


throughput = aRows * 1e6 / (tDone*1e6)
throughputByElements = throughput * dataTensorStream.size(1)
insertion_latency95 = get_latency_percentage(0.95, timestamps)
recall = calculate_recall(gdResults, indexResults)
print("RECALL = "+str(recall))
print("INSERTION LATENCY 95%= "+str(insertion_latency95))
print("QUERY LATENCY = "+str(queryLatency))
print("THROUGHPUT = "+str(throughput))
print("THROUGHPUT BY ELEMENTS = "+str(throughputByElements))
print("ACTIONS = " +str(Actions))