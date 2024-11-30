# OnlineSentimentClassification
## Overview
This application conducts Online Sentiment Classification based on flink, the application looks like this:
 - input: the sentence in data stream
 - output: the sentiment

The DAG is straight forward:
<data stream>-> <encoder> -> <reference knowledge> -> output
Where reference knowledge is using CANDY as Flink plug in.

The model is a multimodal one, but it has never trained for classifyiing text before, so we inject the text knowledge, i.e., labeled sentences, to it by using CANDY without finetuning the model.
The injected text knowledge comes sequentially with verification query in the datastream.
## To run
First, prepare the model by 
```shell
python3 prepareModel.py
``` 
After there is model.pth, run
```shell
python3 main.py
``` 
It will print the classfication accuracy on screen, and also save the processing latency in 1.csv

## To modify
- main.py For change the flink and evaluation settings
- config.csv For change ANNS settings (Only this one is ralted to CANDY)
- prepareModel.py If you want to change the model

