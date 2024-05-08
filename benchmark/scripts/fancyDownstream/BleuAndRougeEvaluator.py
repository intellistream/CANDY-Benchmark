from typing import Callable

import evaluate
import jieba
from loguru import logger
from text2vec import Similarity
from haystack.schema import Document, MultiLabel
from haystack.errors import HaystackError, PipelineError
from haystack.nodes.base import BaseComponent
from typing import List,Tuple,Optional,Dict,Any
from haystack import Pipeline

def catch_all_exceptions(func):
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.warning(repr(e))
    return wrapper


@catch_all_exceptions
def bleu_score(
    continuation: str,
    reference: str,
    with_penalty = False,
    modelPath: Optional[str]="bleu"
) -> float:
    f = lambda text: list(jieba.cut(text))
    bleu = evaluate.load(modelPath)
    results = bleu.compute(predictions=[continuation], references=[[reference]], tokenizer=f)
    
    bleu_avg = results['bleu']
    bleu1 = results['precisions'][0]
    bleu2 = results['precisions'][1]
    bleu3 = results['precisions'][2]
    bleu4 = results['precisions'][3]
    brevity_penalty = results['brevity_penalty']

    if with_penalty:
        return bleu_avg, bleu1, bleu2, bleu3, bleu4
    else:
        return 0.0 if brevity_penalty==0 else bleu_avg/brevity_penalty, bleu1, bleu2, bleu3, bleu4
def bleu_score_list (
        gen:List[str],
        ref:List[str],
        with_penalty = False,
        modelPath: Optional[str]="bleu"
        ) -> float:
    bleu=0
    for i in range(len(gen)):
        bleu_avg, bleu1, bleu2, bleu3, bleu4=bleu_score(gen[i],ref[i],with_penalty,modelPath)
        bleu=bleu+bleu_avg
    return bleu/len(gen)

@catch_all_exceptions
#!pip install rouge_score
def rougeL_score(
    continuation: str,
    reference: str,
    modelPath: Optional[str]="rouge"
) -> float:
    f = lambda text: list(jieba.cut(text))
    rouge = evaluate.load(modelPath)
    results = rouge.compute(predictions=[continuation], references=[[reference]], tokenizer=f, rouge_types=['rougeL'])
    score = results['rougeL']
    return score
def rougeL_score_list(
    gen:List[str],
    ref:List[str],
     modelPath: Optional[str]="rouge"
) -> float:
    ru=0
    for i in range(len(gen)):
        ru=ru+rougeL_score(gen[i],ref[i],modelPath)
    return ru/len(gen)

class EvaluationPrompterTail(BaseComponent):
    """
    The tail node attached to a prompter
    """
    outgoing_edges = 1
    query_count = 0
    index_count = 0
    query_time = 0.0
    index_time = 0.0
    retrieve_time = 0.0
    def __init__(self):
        super().__init__()
        self.collectedAns=[]
        self.collectedPromt=[]
    def run(
        self,
        answers: Optional[List[str]] =None,
        invocation_context: Optional[Dict[str, Any]] = None,
    ) :
        tempStr=answers[0]
        self.collectedAns.append(tempStr)
        tempStr="66666"
        self.collectedPromt.append(invocation_context['prompts'][0])
        print('get ans',answers)
        return {'results':tempStr},tempStr
    def run_batch(
        self,
        queries: Optional[List[str]] = None,
    ):
        ru=[]
        ru2=[]
        for i in queries:
            tempRu,tempRu2=self.run(i)
            ru.append(tempRu['results'])
            ru2.append(tempRu2)
        return {'results':ru},ru2
    def flushRecords(self):
        ru=[]
        ruPrompt=[]
        for i in self.collectedAns:
            ru.append(str(i))
        for i in self.collectedPromt:
            ruPrompt.append(str(i))
        return ru,ruPrompt
    def reset(self):
        self.collectedAns=[]
        self.collectedPromt=[]
class BleuAndRougeEvaluator:
    """
    The Evaluator which outputs bleu and rouge

    Usage: 1. create; 2. call setConfig; 3. call bindPipeline and evaluatePipeline; 4. call getResults; 5. call reset before the next evaluation

    Parameters during config
    bleuBrevityPenalty: whether or not use brevity penalty in bleu calculation, I64, default 0
    bleuModelPath: the path of bleu model, str, default "bleu"
    rougePath: the path of rougle model, str, default "rouge"  
    """
    bleuBrevityPenalty = 0
    bleuModelPath = "bleu"
    rougeModelPath = "rouge"
    def __init__(self):
        print('I can evaluate bleu and rougle')
        self.probTail=EvaluationPrompterTail()
    def reset(self):
        self.probTail.reset()
    def genQueryies(self,keyWords:List[str],queryPrefix:Optional[str]=" What is "):
        ru=[]
        for i in keyWords:
            question=queryPrefix+i+"?"
            ru.append(question)
        return ru
    def setRefAndQueries(self,refs:List[str],keyWords:List[str],queryPrefix:Optional[str]=" What is "):
        """
        set up the reference materials and generate queris by keywords
        """
        self.evaluateQueries=self.genQueryies(keyWords,queryPrefix)
        self.refKnowledge=refs
        self.maxPos = len(refs)
        self.currPos = 0
        self.collectedAns=[]
    def setRefAndRawQueries(self,refs:List[str],queries:List[str]):
        """
        set up the reference materials and generate queris by keywords
        """
        self.evaluateQueries=queries
        self.refKnowledge=refs
        self.maxPos = len(refs)
        self.currPos = 0
        self.collectedAns=[]
    def setConfig(self,cfg:Dict):
        self.myCfg=cfg
        self.bleuBrevityPenalty = self.myCfg.get('bleuBrevityPenalty',0)
        self.bleuModelPath = self.myCfg.get('bleuModelPath','bleu')
        self.rougeModelPath = self.myCfg.get('rougeModelPath','rouge')
    def bindPipeline(self, pipeLineEva:Pipeline,lastStageNames=["Prompter"]):
         pipeLineEva.add_node(component=self.probTail, name= 'EvaTail',inputs= lastStageNames)
    def evaluatePipeline(self,pipeLineEva:Pipeline,annk=1):
        #pipeLineEva.add_node(component=self.probTail, name= 'EvaTail',inputs= lastStageNames)
        for i in self.evaluateQueries:
            pipeLineEva.run(i,params={
        "Retriever": {
            "top_k": annk
        },
        "Reranker": {
            "top_k": annk
        },
        "generation_kwargs":{
            "do_sample": False,
            "max_new_tokens": 128
        }
        })
        print('done query')
    def getResults(self):
        """To reture a Dict containing bleu and rouge scores of both ending answer and intermediate prompt
        """
        collecedRu,collecedPropmt= self.probTail.flushRecords()
        print(collecedRu)
        bleu = bleu_score_list(collecedRu,self.refKnowledge,self.bleuBrevityPenalty,self.bleuModelPath)
        rouge = rougeL_score_list(collecedRu,self.refKnowledge,self.rougeModelPath)
        bleuP = bleu_score_list(collecedPropmt,self.refKnowledge,self.bleuBrevityPenalty,self.bleuModelPath)
        rougeP = rougeL_score_list(collecedPropmt,self.refKnowledge,self.rougeModelPath)
        return {"bleu_answer":bleu,"rouge_answer":rouge,"bleu_prompt":bleuP,"rouge_promt":rougeP}