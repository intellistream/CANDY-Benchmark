import os
import multimodal 

def main():
    # first, install python deps
    os.system('python3 -m pip install -r deps.txt')
    # next, download corpus
    exeSpace = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/"
    files = 0
    checkList = ['datasets/coco/query_image.fvecs',
      'datasets/coco/query_captions.fvecs',
      'datasets/coco/query_shuffle.fvecs',
     'datasets/coco/query_shuffle.fvecs',
    'datasets/coco/data_image.fvecs',
      'datasets/coco/data_captions.fvecs',
      'datasets/coco/data_shuffle.fvecs',
     'datasets/coco/data_shuffle.fvecs',
    ]
    for i in checkList:
        if(os.path.exists(exeSpace+i)):
            files = files +1
    if(files >= len(checkList)) :
        print('skip generation of coco')
    else:
        multimodal.main()


if __name__ == "__main__":
    main()
