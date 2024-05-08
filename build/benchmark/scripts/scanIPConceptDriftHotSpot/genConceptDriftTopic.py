import testNltk as testNltk
from warthunder2 import warthunderRead
from dpr_synthetic import generate_dpr_embeddings_with_custom_sentences
import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, AutoTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import BatchEncoding
import os
import random
import numpy as np
import numpy.typing as npt
import nltk
import sys
from typing import Optional, Union
from typing import Dict, List, Optional, Union, Any
import time, pickle


def save_as_fvecs(tensor, fname):
    # Convert PyTorch tensor to NumPy array
    # embeddings = np.memmap(filename, dtype='float32', mode='w+', shape=(tensor.size(0),tensor.size(1)))
    embeddings = tensor.numpy()
    n, d = embeddings.shape
    m1 = np.memmap(fname, dtype='int32', mode='w+', shape=(n, d + 1))
    m1[:, 0] = d
    m1[:, 1:] = embeddings.view('int32')


class warthunderEncoder():
    def __init__(self) -> None:
        self.ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        self.ctx_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        self.device = "cuda:" + str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
        print(self.device)
        self.ctx_encoder = self.ctx_encoder.to(self.device)
        self.q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        self.q_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        self.q_encoder = self.q_encoder.to(self.device)

    def generate_embeddings(self, model: Union[DPRContextEncoder, DPRQuestionEncoder], encoded_input: BatchEncoding,
                            dim: int,
                            batch_size: int, device: str) -> torch.tensor:
        n_seq = len(encoded_input['input_ids'])
        shapeOut = (n_seq, dim)
        token_embeddings_out = torch.zeros(shapeOut)

        print('Doing inference for', n_seq, 'sequences.')

        model.eval()

        num_batches = int(np.ceil(float(n_seq) / batch_size))
        batch_print = 100
        if device != "cpu":
            start1 = torch.cuda.Event(enable_timing=True)
            end1 = torch.cuda.Event(enable_timing=True)
            start1.record()
        with torch.no_grad():
            for batch in range(num_batches):

                batch_init = batch * batch_size
                batch_end = np.min([batch_init + batch_size, n_seq])

                token_embeddings = model(encoded_input['input_ids'][batch_init:batch_end].to(device),
                                         encoded_input['attention_mask'][batch_init:batch_end].to(device))
                token_embeddings_out[batch_init:batch_end, :] = token_embeddings.pooler_output.cpu()
                if not (batch % batch_print):
                    print('Doing inference for batch', batch, 'of', num_batches)
        if device != "cpu":
            end1.record()
            torch.cuda.synchronize()
            print(f'Inference for {n_seq}, sequences took {(start1.elapsed_time(end1) / 1000):.2f} s')

        return token_embeddings_out

    def tokenize_texts(self, ctx_tokenizer: AutoTokenizer, texts: list, max_length: Optional[int] = None,
                       doc_stride: Optional[int] = None,
                       text_type: Optional[str] = "context", save_sentences: Optional[bool] = False, \
                       fname_sentences: Optional[str] = None) -> BatchEncoding:
        if text_type == "context":
            if max_length == None:
                max_length = 8192
                # print("Setting max_length to", max_length)
            if doc_stride == None:
                doc_stride = int(max_length / 2)
                # print("Setting doc_stride to", doc_stride)

        start = time.time()
        if text_type == "context":
            encoded_inputs = ctx_tokenizer(texts, padding=True, truncation=True, max_length=max_length, \
                                           return_overflowing_tokens=True, \
                                           stride=doc_stride, return_tensors="pt")
        elif text_type == "query":
            encoded_inputs = ctx_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        end = time.time()
        delta_time = end - start
        # print(f'Tokenization for {len(texts)}, contexts took {delta_time:.2f} s')

        n_seq = len(encoded_inputs['input_ids'])
        if save_sentences:
            if fname_sentences is not None:
                # Code to generate sentences from tokens
                sentences = []
                for i in range(n_seq):
                    # print('Processing sentence', i, 'of', n_seq)
                    sentences += [' '.join(encoded_inputs.tokens(i))]

                with open(fname_sentences, 'wb') as f:
                    pickle.dump(sentences, f)
                del sentences
            else:
                raise BaseException(
                    'tokenize_texts: The filename where the original sentences will be saved was not specified.')

        return encoded_inputs

    def encodeContext(self, ctx: str, batchSize: Optional[int] = 64):
        encoded_contexts = self.tokenize_texts(self.ctx_tokenizer, ctx, text_type="context").to(self.device)
        embeddings_batch = self.generate_embeddings(self.ctx_encoder, encoded_contexts, 768, batchSize,
                                                    self.device)
        # Define the string to initialize each element
        initial_string = ctx

        # Create the list using a list comprehension
        list_of_strings = [initial_string for _ in range(embeddings_batch.size(0))]
        return embeddings_batch, list_of_strings

    def encodeQuery(self, qtx: str, batchSize: Optional[int] = 64):
        text_type = "query"
        encoded_quries = self.tokenize_texts(self.q_tokenizer, qtx, text_type=text_type).to(self.device)
        embeddings_batch = self.generate_embeddings(self.q_encoder, encoded_quries, 768, batchSize,
                                                    self.device)
        initial_string = qtx
        # Create the list using a list comprehension
        list_of_strings = [initial_string for _ in range(embeddings_batch.size(0))]
        return embeddings_batch, list_of_strings

    def ctxListToFvecs(self, ctxList: List[List[str]], fname: str):
        starPosVec = []
        endPosVec = []
        currStart = 0
        for i in ctxList:
            starPosVec.append(currStart)
            currStart += len(i)
            endPosVec.append(currStart)
        ruTensor = torch.zeros(currStart, 768)
        startPos = 0
        endPos = 0
        for i in range(len(ctxList)):
            sentenceI = ctxList[i]
            ru, rs = self.encodeContext(sentenceI)
            # print(ru)
            ruTensor[starPosVec[i]:endPosVec[i], :] = ru[0:endPosVec[i] - starPosVec[i], :]
        save_as_fvecs(ruTensor, fname)
        print(ruTensor, ruTensor.size())

    def qtxListToFvecs(self, qtxList: List[List[str]], fname: str):
        starPosVec = []
        endPosVec = []
        currStart = 0
        for i in qtxList:
            starPosVec.append(currStart)
            currStart += len(i)
            endPosVec.append(currStart)
        ruTensor = torch.zeros(currStart, 768)
        startPos = 0
        endPos = 0
        for i in range(len(qtxList)):
            sentenceI = qtxList[i]
            ru, rs = self.encodeQuery(sentenceI)
            # print(ru)
            ruTensor[starPosVec[i]:endPosVec[i], :] = ru[0:endPosVec[i] - starPosVec[i], :]
        save_as_fvecs(ruTensor, fname)
        return ruTensor


def genConceptDriftTopic(topic0Len=50000, topic1Len=50000, topic0Prefix="USA_aircraft",
                         topic1Prefix="USSR_ground_vehicles"):
    k0, d0, t0, h0 = warthunderRead.paraseInCategories([topic0Prefix])
    k1, d1, t1, h1 = warthunderRead.paraseInCategories([topic1Prefix])
    topic0 = k0
    topic1 = k1
    nouns, verbs = testNltk.generate_dictionares(100)
    sentences0, q0 = testNltk.generate_sentences_with_pollution(topic0, verbs, topic0Len, '', 0)
    sentences1, q1 = testNltk.generate_sentences_with_pollution(topic1, verbs, topic1Len, '', 0)
    return [sentences0, sentences1], [q0, q1]


def genConceptDriftHotSpot(topic0Len=50000, topic1Len=50000, hotspotFreq=0.1, topic0Prefix="USA_aircraft",
                           hotSpotWord="T-34-85"):
    k0, d0, t0, h0 = warthunderRead.paraseInCategories([topic0Prefix])
    k1, d1, t1, h1 = warthunderRead.paraseInCategories([topic0Prefix])
    topic0 = k0
    topic1 = k1
    nouns, verbs = testNltk.generate_dictionares(100)
    sentences0, q0 = testNltk.generate_sentences_with_pollution(topic0, verbs, topic0Len, '', 0)
    sentences1, q1 = testNltk.generate_sentences_with_pollution(topic0, verbs, topic1Len, hotSpotWord, hotspotFreq)
    return [sentences0, sentences1], [q0, q1]


def genConceptDriftTopicEmbeddings(topic0Len=50000, topic1Len=50000, outputFname='wt.fvecs', outputQname='wtq.fvecs',
                                   topic0Prefix="USA_aircraft", topic1Prefix="USSR_ground_vehicles"):
    ru, queries = genConceptDriftTopic(topic0Len, topic1Len)
    os.system('rm *.fvecs')
    wte = warthunderEncoder()
    # print(ru,queries)
    print(len(ru), len(queries))
    wte.ctxListToFvecs(ru, outputFname)
    wte.qtxListToFvecs(queries, outputQname)


def genConceptDriftHotSpotEmbeddings(topic0Len=50000, topic1Len=50000, hotspotFreq=0.1, outputFname='wt.fvecs',
                                     outputQname='wtq.fvecs', topic0Prefix="USA_aircraft", hotSpotWord="T-34-85"):
    ru, queries = genConceptDriftHotSpot(topic0Len, topic1Len, hotspotFreq)
    os.system('rm *.fvecs')
    wte = warthunderEncoder()
    print(ru, queries)
    # print(len(ru),len(queries))
    wte.ctxListToFvecs(ru, outputFname)
    wte.qtxListToFvecs(queries, outputQname)
    # generate_dpr_embeddings_with_custom_sentences(768,64,ruList,embeddingSizes=[topic0Len,topic1Len],outputFname=outputFname,generate_queries=True,outputQname=outputQname)


if __name__ == "__main__":
    print('666')
    genConceptDriftTopicEmbeddings(10, 10)
    print('999')
    genConceptDriftHotSpotEmbeddings(10, 10, 0.5)
