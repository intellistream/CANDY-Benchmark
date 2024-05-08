# DPR Dataset Generator

This repository provides code to generate base and query vector datasets for similarity search benchmarking and
evaluation on high-dimensional vectors stemming from large language models.
With the dense passage retriever (DPR) [[1]](#1), we encode text snippets from the C4 dataset [[2]](#2) to generate
768-dimensional vectors:

- context DPR embeddings for the base set and
- question DPR embeddings for the query set.

The metric for similarity search is inner product [[1]](#1).

The number of base and query embedding vectors is parametrizable.

## DPR10M

A specific instance with 10 million base vectors and 10,000 query vectors is introduced in [[3]](#3). Use the
script [dpr_dataset_10M.py](dpr_dataset_10M.py) to generate this dataset. The corresponding ground-truth (
available [here](gtruth_dpr10M_innerProduct.ivecs)) is generated conducting an exhaustive search with the inner product
metric.

Here is a summary of the **steps to generate this dataset**:

1. **Download the files** corresponding to the `en` variant of the C4 dataset
   accesible [here](https://huggingface.co/datasets/allenai/c4).
   The complete set of files requires 350GB of storage, so you might want to follow the instructions to download only a
   subset. For example, to generate 10M embeddings
   we used the first 2 files from the train set (i.e., files `c4-train.00000-of-01024.json.gz`
   and `c4-train.00001-of-01024.json.gz` in `c4/en/train`).

2. **Execute** the `generate_dpr_embeddings` function to generate a `.fvecs` file containing the new embeddings.
   Note that different settings should be used to generate the **base vectors** and the **query set**, as they use the
   DPR context and query encoders respectively.
   See the script [dpr_dataset_10M.py](dpr_dataset_10M.py) for details.

```
# Example code to generate base vectors

from dpr_dataset_generate import generate_dpr_embeddings

base_C4_folder = '/home/username/research/datasets/c4/en'  # Set this path to where your c4/en folder is located
cache_folder = f'/home/username/.cache/huggingface/datasets/'  # Set to the hugginface datasets cache path
dataset_dir = f'{base_C4_folder}/train/'

num_embd = 10000000
init_file = 0
num_of_files = 2  # Make sure the input files (2 in this case) are enough to generate the 
                  # requested number  of embeddings. 
                  # To get an estimate, use the optional parameter get_total_embeddings_only 
                  # to get the number of embeddings that can be generated from a certain 
                  # group of files without actually generating the embeddings.
fname_prefix_out = 'c4-en'
doc_stride = 32
max_length = 64
batch_size = 512
dim = 768

generate_dpr_embeddings(init_file, num_of_files, num_embd, doc_stride, max_length, dim,
                        batch_size,
                        dataset_dir, fname_prefix_out, cache_folder)
```

3. **Generate the ground-truth** by conducting an exhaustive search with the inner product metric.
   We provide the [ground-truth](gtruth_dpr10M_innerProduct.ivecs) for the dataset generated using
   [dpr_dataset_10M.py](dpr_dataset_10M.py).

> **_NOTE:_**  Due to floating-point arithmetic precision the vector embeddings generated using the provided
> code in different machines may slightly vary. Keep in mind that this could cause small discrepancies with the provided
> ground-truth.

4. Functions `read_fvecs` and `read_ivecs` can be used to read `.fvecs` and `.ivecs` files respectively.

## References

Reference to cite when you use datasets generated with this code in a research paper:

```
@article{aguerrebere2023similarity,
        title={Similarity search in the blink of an eye with compressed indices},
        volume = {16},
        number = {11},    
        pages = {3433--3446},    
        journal = {Proceedings of the VLDB Endowment},
        author={Cecilia Aguerrebere and Ishwar Bhati and Mark Hildebrand and Mariano Tepper and Ted Willke},        
        year = {2023}
}
```

<a id="1">[1]</a>
Karpukhin, V.; Oguz, B.; Min, S.; Lewis, P.; Wu, L.; Edunov, S.; Chen, D.; Yih, W..: Dense Passage
Retrieval for Open-Domain Question Answering. In: Proceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing (EMNLP). 6769–6781. (2020)

<a id="2">[2]</a>
Raffel, C.; Shazeer, N.; Roberts, A.; Lee, K.; Narang, S.; Matena, M.; Zhou, Y.; Li, W.; Liu,
P.J.: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.
In: The Journal of Machine Learning Research 21,140:1–140:67.(2020)

<a id="3">[3]</a>
Aguerrebere, C.; Bhati I.; Hildebrand M.; Tepper M.; Willke T.:Similarity search in the blink of an eye with compressed
indices. In: Proceedings of the VLDB Endowment, 16, 11, 3433 - 3446. (2023)

This "research quality code"  is for Non-Commercial purposes provided by Intel "As Is" without any express or implied
warranty of any kind. Please see the dataset's applicable license for terms and conditions. Intel does not own the
rights to this data set and does not confer any rights to it. Intel does not warrant or assume responsibility for the
accuracy or completeness of any information, text, graphics, links or other items within the code. A thorough security
review has not been performed on this code. Additionally, this repository may contain components that are out of date or
contain known security vulnerabilities.
