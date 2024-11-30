import torch
from torch.utils.data import DataLoader
from datasets import load_dataset,concatenate_datasets
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import MapFunction,SinkFunction
from pyflink.table import StreamTableEnvironment, EnvironmentSettings

# Assuming your model script is named classifier.py
from classifier import MultimodalSentimentClassifier
import time,csv,json,random
import PyCANDY as candy
# Step 1: Load IMDB dataset
def loadImdbDataset(trainSize=5000,testSize=500):
    dataset = load_dataset("imdb")
    print(len(dataset['train']))
    print(len(dataset['test']))
    trainDataset = dataset['train'].select(range(trainSize))
    testDataset = dataset['test'].select(range(testSize))
    return trainDataset, testDataset
def mixData(a, b, begin, replace_probability):
    """
    Mixes two lists `a` and `b` into a new list.
    
    Args:
        a (list): First dataset in the form [(text, label)].
        b (list): Second dataset in the form [(text, label)].
        begin (int): Number of initial elements to copy from `a`.
        replace_probability (float): Probability of replacing an element of `a` with `b`.
    
    Returns:
        list: Mixed dataset.
    """
    if len(a) != len(b):
        raise ValueError("Input lists `a` and `b` must have the same length.")

    mixed_data = []
    ref_data = []
    # Step 1: Take the first `begin` elements from `a`.
    mixed_data.extend(a[:begin])
    ref_data.extend(a[:begin])
    # Step 2: For the rest, replace `a[i]` with `b[i]` with the given probability.
    for i in range(begin, len(a)):
        if random.random() < replace_probability:
            # Replace a[i] with b[i], set label to -1
            mixed_data.append((b[i][0], -1))
            #print(f'label={b[i][1]}')
            ref_data.append(b[i])
        else:
            # Keep a[i] as is
            mixed_data.append(a[i])
            ref_data.append(a[i])
    return mixed_data,ref_data
def generateRefCsv(refStream,exeStream,output_path):
    with open(output_path, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # Write the header
            csv_writer.writerow(['latencyEmbedding','latencyKB','label',"writeKB"])
            for i in range(len(refStream)):
                writeKB= 1
                emb,rw= exeStream[i]
                emb,label = refStream[i]
                if rw==-1:
                     writeKB= 0
                     #print(f"This is a query, with label {label}")
                csv_writer.writerow([0,0,label,writeKB])
# Step 2: Preprocess data
def preprocessData(dataset):
    return [(example['text'], (example['label'])) for example in dataset]
# Step 3: PyFlink pipeline
# Loading the model's state_dict
def setSeed(seed):
    torch.manual_seed(seed)               # Set seed for PyTorch
    torch.cuda.manual_seed(seed)          # If you're using a GPU
    torch.cuda.manual_seed_all(seed)      # For multi-GPU setups
def loadModel(model, file_path="model.pth"):
    model.load_state_dict(torch.load(file_path))
    model.eval()  # Set the model to evaluation mode after loading
    print(f"Model loaded from {file_path}")
    return model
class SentenceEmbedder(MapFunction):
    def __init__(self, modelPath, device='cpu'):
        self.device = device
    def open(self,runtime_context):
        model = MultimodalSentimentClassifier(device=self.device)
        self.model= loadModel(model,'model.pth')
        self.model.eval()
    def map(self, value):
        text, label = value
        startTime = time.time()  # Record the start time
        # Convert text to tensor using the model
        embedding = self.model.convert_text_to_tensor(text)
        endTime = time.time()  # Record the end time
        latency = endTime - startTime  # Calculate latency in seconds
        #print(f'label={label}')
        return {
            "embedding": embedding.cpu().numpy().tolist(),
            "label": label,
            "latencyEmbedding": latency  # Add latency to the output
        }
class CandyKnowledgeBase(MapFunction):
    def __init__(self, k=3,  dict=None):
        self.k = k
        self.cfgDict = dict
       
    def open(self, runtime_context):
        """
        Initialize non-picklable objects.
        """
        idxTag = self.cfgDict['indexTag']
        self.indexPtr0 = candy.createIndex(idxTag)
        self.indexPtr1 = candy.createIndex(idxTag)
        print('chose index: '+idxTag)
        self.configMap = candy.dictToConfigMap(self.cfgDict)
        self.indexPtr0.setConfig(self.configMap)
        self.indexPtr1.setConfig(self.configMap)
        self.workerId = runtime_context.get_index_of_this_subtask()
    def map(self, value):
        """
        Add data to the buffer and perform local kNN if the buffer is full.
        """
        tensorTemp = torch.tensor(value['embedding'])
        tensorTemp = torch.reshape(tensorTemp, (1, -1))
        label = value['label']
        startTime = time.time()  # Record the start time
        writeKB= 1
        predictionLable = label
        if(label==0):
            self.indexPtr0.insertTensor(tensorTemp)  # Insert tensor into the index0
        if(label==1):
            self.indexPtr1.insertTensor(tensorTemp)  # Insert tensor into the index1
        if(label==-1):
            #print('I should predict')
            knnResults0 = self.indexPtr0.searchTensor(tensorTemp, self.k)[0]
            knnResults1 = self.indexPtr1.searchTensor(tensorTemp, self.k)[0]
            ipTo0 = torch.matmul(knnResults0,tensorTemp.T)
            meanIpTo0 = torch.mean(ipTo0)
            ipTo1 = torch.matmul(knnResults1,tensorTemp.T)
            meanIpTo1 = torch.mean(ipTo1)
            if(meanIpTo0>meanIpTo1):
                predictionLable = 0
            else:
                predictionLable = 1
            writeKB= 0
        endTime = time.time()  # Record the end time
        latency = endTime - startTime  # Calculate latency in seconds
        return {
            "embedding":value['embedding'],
            "label": predictionLable,
            "latencyKB":latency,  # Add latency to the output
            "latencyEmbedding": value['latencyEmbedding'],  # Add latency to the output
            "writeKB": writeKB
        }
# LatencyCollector as MapFunction
class LatencyCollector(MapFunction):
    def __init__(self,output_path='1.csv'):
        self.latency_data = []
        self.outputPath = output_path
    def open(self, runtime_context):
        # Initialize the data structure to collect latency and label information
        self.latency_data = []

    def map(self, value):
        # Collect latency and label in memory
        self.latency_data.append({
            "latencyEmbedding": value['latencyEmbedding'],
            "latencyKB":value['latencyKB'],
            "label": value['label'],
            "writeKB": value['writeKB']
        })
        return value  # Pass data along if needed for further processing

    def close(self):
        # Export latency data to a JSON file when the pipeline ends
        with open(self.outputPath, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # Write the header
            csv_writer.writerow(['latencyEmbedding','latencyKB','label',"writeKB"])
            # Write the rows
            for record in self.latency_data:
                csv_writer.writerow([record['latencyEmbedding'],record['latencyKB'],record['label'],record['writeKB']])
        print(f"Latency data saved to '{self.outputPath}'.")
def flinkPipeline(data):
    # Set up the PyFlink environment
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    # Convert data to a PyFlink data stream
    stream = env.from_collection(data)

    # Apply the SentenceEmbedder function
    embeddingMapFunction = SentenceEmbedder(modelPath="path/to/your/model", device="cuda:0" if torch.cuda.is_available() else "cpu")
    embeddedStream = stream.map(embeddingMapFunction)
    configMap = candy.ConfigMap()
    configMap.fromFile('config.csv')
    cfg = candy.configMapToDict(configMap)
    #cfg={'vecDim':768,'metricType':"IP"}
    knowledgeBaseFunction = CandyKnowledgeBase(k=10,dict=cfg)
    kbStream = embeddedStream.map(knowledgeBaseFunction)
    # Apply CsvWriter sink
     # Set up the collector
    collector = LatencyCollector(output_path='1.csv')
    # Apply a lambda function to collect results
    result=kbStream.map(collector)
    env.execute("Multimodal Sentiment Embedding Pipeline")
    
def calculateAccuracy(file_a, file_b, write_kb_column='writeKB', label_column='label'):
    """
    Compare two CSV files row by row based on the 'writeKB' column and the 'label' column.

    Args:
        file_a (str): Path to the first CSV file.
        file_b (str): Path to the second CSV file.
        write_kb_column (str): Name of the column in file_a to check for 0.
        label_column (str): Name of the column to compare for correctness.

    Returns:
        float: Accuracy as correct/count.
    """
    count = 0
    correct = 0

    with open(file_a, mode='r') as csv_a, open(file_b, mode='r') as csv_b:
        reader_a = csv.DictReader(csv_a)
        reader_b = csv.DictReader(csv_b)

        # Ensure both files have the same structure
        if reader_a.fieldnames != reader_b.fieldnames:
            raise ValueError("CSV files must have the same structure and column names.")

        # Traverse rows in both CSVs
        for row_a, row_b in zip(reader_a, reader_b):
            if int(row_a[write_kb_column]) == 0:
                count += 1
                if row_a[label_column] == row_b[label_column]:
                    correct += 1

    # Calculate and return accuracy
    return correct / count if count > 0 else 0.0
def main():
    # Load and preprocess IMDB dataset
    insertDataset, queryDataset = loadImdbDataset(trainSize=1000,testSize=1000)
    setSeed(999)
    #insertDataset=[{'text':"hello",'label':1},{'text':"fuck",'label':0},{'text':"hi",'label':-1}]
    insertStream = preprocessData(insertDataset)
    queryStream = preprocessData(queryDataset)
    exeStream,refStream = mixData(insertStream,queryStream,500,0.5)
    generateRefCsv(refStream,exeStream,'groundTruth.csv')
    print('data is prepared')
    flinkPipeline(exeStream)
    acc= calculateAccuracy('groundTruth.csv','1.csv')
    print(f'accuracy={acc}')
    #print(insertStream)
if __name__ == "__main__":
    main()