import os
import nltk

def testC4Corpus():
    if (os.path.exists('c4/c4done')):
        print('corpus is done')
    else:
        os.system('./downloadC4Text.sh')


def gen100KDPR():
    exeSpace = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/"
    filePath = exeSpace + 'datasets/DPR/DPR100KC4.fvecs'
    if (os.path.exists(filePath)):
        print('100K embedding is done')
    else:
        import dpr_dataset_100K
        os.system("mkdir " + exeSpace + "datasets/")
        os.system("mkdir " + exeSpace + "datasets/DPR")
        dpr_dataset_100K.main()
        os.system("cp c4/en/embeddings/c4-train_base_0M_files0_1.fvecs " + filePath)
        queryPath = exeSpace + 'datasets/DPR/DPR10KC4Q.fvecs'
        os.system("cp c4/en/embeddings/c4-validation_queries_10k_files0_1.fvecs " + queryPath)


def gen10MDPR():
    exeSpace = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/"
    filePath = exeSpace + 'datasets/DPR/DPR10MKC4.fvecs'
    if (os.path.exists(filePath)):
        print('100K embedding is done')
    else:
        import dpr_dataset_10M
        dpr_dataset_10M.main()
        os.system("cp c4/en/embeddings/c4-train_base_10M_files0_1.fvecs " + filePath)


def main():
    # first, install python deps
    os.system('python3 -m pip install -r deps.txt')
    nltk.download('wordnet')
    # next, download corpus
    testC4Corpus()
    # NEXT, GEN 100K
    gen100KDPR()
    # gen10MDPR()


if __name__ == "__main__":
    main()
