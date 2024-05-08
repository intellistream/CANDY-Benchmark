# This is to generate a synthetic dataset using dpr style
# The Data base is simply embedding of <noun A> <verb> <noun B>, where we introduce the shifed frequency of <noun A>
import testNltk
import random
import torch
from typing import Optional
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, AutoTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import BatchEncoding
import os
from dpr_dataset_generate import tokenize_texts, generate_embeddings, random_sample_tensor, fvecs_write_from_mmap, \
    append_fvecs
import numpy as np


def append_tensors(tensor1, tensor2, dim=0):
    """
    Appends one PyTorch tensor to another along a specified dimension.

    Parameters:
    - tensor1 (torch.Tensor): First tensor.
    - tensor2 (torch.Tensor): Second tensor to be appended to the first.
    - dim (int): Dimension along which to concatenate the tensors. Default is 0.

    Returns:
    - torch.Tensor: Concatenated tensor.

    """
    return torch.cat((tensor1, tensor2), dim=dim)


def generate_dpr_embeddings_with_synthetic_context_shift(dim: int, batch_size: int, dictionary_size: int, words,
                                                         frequencies, sentences, embeddingSizes, outputFname: str,
                                                         generate_queries: Optional[bool] = False,
                                                         outputQname: Optional[str] = 'query.fvecs'):
    """
    Generate DPR embeddings from text with synthetic context shift 

    Keyword arguments:
    
    dim -- Dimensionality of the generated embeddings, it is defined by the model.
    batch_size -- Batch size used at inference time to generate the embeddings.
    words  -- The highlight words to be shown in the sentence context, in list of str
    frequencies  -- The frequency of each highlighted words to be shown in the sentence context, in list of float
    sentences  -- The number of highlighted sentences for each highlighted word, in list of int
    embeddingSizes  -- The number to restrict generated embeddings related with each word, in list of int
    outputFname -- The name of outputfile
    generate_queries -- whether or not generate queries
    outputQname -- the output name of query
    """
    random_nouns, random_verbs = testNltk.generate_dictionares(dictionary_size)
    print('Dictionary is generated, size=' + str(dictionary_size))
    contexts = []
    queries = []
    totalEmbeddings = 0
    for i in range(len(words)):
        datasets = testNltk.generate_sentences_with_pollution(random_nouns, random_verbs, sentences[i], words[i],
                                                              frequencies[i])
        contexts.append(datasets)
        totalEmbeddings = totalEmbeddings + embeddingSizes[i]
    if os.path.isfile(outputFname):
        print(f'File {outputFname} already exists. Overwrite it!')
        os.remove(outputFname)
    if os.path.isfile(outputQname):
        print(f'File {outputQname} already exists. Overwrite it!')
        os.remove(outputQname)
        # sys.exit()
    embeddings = np.memmap(outputFname, dtype='float32', mode='w+', shape=(totalEmbeddings, dim))
    curr_total_emb = 0
    init_emb = 0
    """
    0. prepare the encoder
    """
    ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    ctx_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    device = "cuda:" + str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
    ctx_encoder = ctx_encoder.to(device)
    # embedingTensor = None
    # q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    # q_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    """
    1. generate the token and embedding for each context
    """
    text_type = "context"
    for i in range(len(words)):
        ctx = contexts[i]
        encoded_contexts = tokenize_texts(ctx_tokenizer, ctx, text_type=text_type).to(device)
        fname_mmap_aux = "temp1.mmap"
        if (len(encoded_contexts['input_ids']) > embeddingSizes[i]):
            print('Too many embeddings at ' + words[i] + ", automatically remove some")
            # Generate random indices for sampling
            encoded_contexts['input_ids'] = random_sample_tensor(encoded_contexts['input_ids'], embeddingSizes[i])
            encoded_contexts['attention_mask'] = random_sample_tensor(encoded_contexts['attention_mask'],
                                                                      embeddingSizes[i])
            # print(len(encoded_contexts))

        embeddings_batch = generate_embeddings(ctx_encoder, encoded_contexts, dim, batch_size, fname_mmap_aux,
                                               device)

        del encoded_contexts
        curr_total_emb += embeddings_batch.shape[0]
        print(curr_total_emb, 'embeddings generated so far,', embeddings_batch.shape[0])
        if curr_total_emb < totalEmbeddings:
            # keep all generated embeddings
            end_emb = init_emb + embeddings_batch.shape[0]
            embeddings[init_emb:end_emb] = embeddings_batch
        else:
            # complete the last embeddings and finish
            embeddings[init_emb:] = embeddings_batch[:(totalEmbeddings - init_emb)]

        init_emb += embeddings_batch.shape[0]
    fvecs_write_from_mmap(outputFname, embeddings[:curr_total_emb])
    os.remove(fname_mmap_aux)
    print('embedding for context is done')
    """
    2. generate the token and embedding for each query
    """
    # embedingTensor = None
    del ctx_encoder
    del ctx_tokenizer
    q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    q_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    q_encoder = q_encoder.to(device)
    embeddings = np.memmap(outputQname, dtype='float32', mode='w+', shape=(totalEmbeddings, dim))
    if (generate_queries == False):
        return
    for i in range(len(words)):
        questions = testNltk.sentences_to_questions(contexts[i])
        queries.append(questions)
    curr_total_emb = 0
    init_emb = 0
    del contexts
    text_type = "query"
    for i in range(len(words)):
        qtx = queries[i]
        encoded_quries = tokenize_texts(q_tokenizer, qtx, text_type=text_type).to(device)
        fname_mmap_aux = "temp1.mmap"
        if (len(encoded_quries['input_ids']) > embeddingSizes[i]):
            print('Too many embeddings at ' + words[i] + ", automatically remove some")
            # Generate random indices for sampling
            encoded_quries['input_ids'] = random_sample_tensor(encoded_quries['input_ids'], embeddingSizes[i])
            encoded_quries['attention_mask'] = random_sample_tensor(encoded_quries['attention_mask'], embeddingSizes[i])
            # print(len(encoded_contexts))

        embeddings_batch = generate_embeddings(q_encoder, encoded_quries, dim, batch_size, fname_mmap_aux,
                                               device)

        del encoded_quries
        curr_total_emb += embeddings_batch.shape[0]
        print(curr_total_emb, 'embeddings generated so far,', embeddings_batch.shape[0])
        if curr_total_emb < totalEmbeddings:
            # keep all generated embeddings
            end_emb = init_emb + embeddings_batch.shape[0]
            embeddings[init_emb:end_emb] = embeddings_batch
        else:
            # complete the last embeddings and finish
            embeddings[init_emb:] = embeddings_batch[:(totalEmbeddings - init_emb)]

        init_emb += embeddings_batch.shape[0]
    fvecs_write_from_mmap(outputQname, embeddings[:curr_total_emb])
    os.remove(fname_mmap_aux)
    print('embedding for queries is done')
    return


if __name__ == "__main__":
    random.seed(42)
    words = ['Covid19', 'Disinfect']
    frequencies = [0.5, 0.5]
    sentences = [50000, 50000]
    generate_dpr_embeddings_with_synthetic_context_shift(768, 512, 10, words, frequencies, sentences, sentences,
                                                         "DPRSYN100K.fvecs", True, "DPRSYN100KQ.fvecs")
