import os
import multimodal 

def main():
    # first, install python deps
    os.system('python3 -m pip install -r deps.txt')
    # next, download corpus
    multimodal.main()


if __name__ == "__main__":
    main()
