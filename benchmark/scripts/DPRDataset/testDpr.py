from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, AutoTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import BatchEncoding
import torch
import time
import os, shutil
import gzip
from datasets import load_dataset
# the files used/created by pickle are temporary and don't pose any security issue
import pickle  # nosec
import random
import numpy as np
import numpy.typing as npt
import nltk
import sys
from typing import Optional, Union


# list of text into tokens
def tokenize_texts(ctx_tokenizer: AutoTokenizer, texts: list, max_length: Optional[int] = None,
                   doc_stride: Optional[int] = None,
                   text_type: Optional[str] = "context", save_sentences: Optional[bool] = False, \
                   fname_sentences: Optional[str] = None) -> BatchEncoding:
    if text_type == "context":
        if max_length == None:
            max_length = 64
            print("Setting max_length to", max_length)
        if doc_stride == None:
            doc_stride = int(max_length / 2)
            print("Setting doc_stride to", doc_stride)

    start = time.time()
    if text_type == "context":
        encoded_inputs = ctx_tokenizer(texts, padding=True, truncation=True, max_length=max_length, \
                                       return_overflowing_tokens=True, \
                                       stride=doc_stride, return_tensors="pt")
    elif text_type == "query":
        encoded_inputs = ctx_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    end = time.time()
    delta_time = end - start
    print(f'Tokenization for {len(texts)}, contexts took {delta_time:.2f} s')

    n_seq = len(encoded_inputs['input_ids'])
    if save_sentences:
        if fname_sentences is not None:
            # Code to generate sentences from tokens
            sentences = []
            for i in range(n_seq):
                if not (i % 100000):
                    print('Processing sentence', i, 'of', n_seq)
                sentences += [' '.join(encoded_inputs.tokens(i))]

            with open(fname_sentences, 'wb') as f:
                pickle.dump(sentences, f)
            del sentences
        else:
            raise BaseException(
                'tokenize_texts: The filename where the original sentences will be saved was not specified.')

    return encoded_inputs


# tokens into embeddings and save
def generate_embeddings(model: Union[DPRContextEncoder, DPRQuestionEncoder], encoded_input: BatchEncoding, dim: int,
                        batch_size: int, fname_mmap: str, device: str) -> np.memmap:
    n_seq = len(encoded_input['input_ids'])
    token_embeddings_out = np.memmap(fname_mmap, dtype='float32', \
                                     mode='w+', shape=(n_seq, dim))

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
            token_embeddings_out[batch_init:batch_end, :] = token_embeddings.pooler_output.cpu().numpy()

            del token_embeddings

            if not (batch % batch_print):
                print('Doing inference for batch', batch, 'of', num_batches)
    if device != "cpu":
        end1.record()
        torch.cuda.synchronize()
        print(f'Inference for {n_seq}, sequences took {(start1.elapsed_time(end1) / 1000):.2f} s')

    return token_embeddings_out


def main():
    print('hello world')
    save_sentences = True
    fname_sentences_query = '1.pkl'
    fname_mmap_aux = '1.mmap'
    dim = 768
    batch_size = 10

    ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    ctx_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    q_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    text_type = "query"
    queries = ['hello world', 'hi world', 'hello word']
    device = "cuda:" + str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
    ctx_encoder = ctx_encoder.to(device)
    encoded_queries = tokenize_texts(ctx_tokenizer, queries, text_type=text_type, \
                                     save_sentences=save_sentences, fname_sentences=fname_sentences_query).to(device)
    print('Generating embeddings for queries.')

    embeddings_batch = generate_embeddings(ctx_encoder, encoded_queries, dim, batch_size, fname_mmap_aux,
                                           device)
    print(embeddings_batch)


if __name__ == "__main__":
    main()
