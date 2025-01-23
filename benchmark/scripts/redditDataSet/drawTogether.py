import os
import redditDataset 

def main():
    # first, install python deps
    os.system('python3 -m pip install -r deps.txt')
    # next, download corpus
    redditDataset.main()


if __name__ == "__main__":
    main()
