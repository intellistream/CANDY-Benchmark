import os


def main():
    # first, install python deps
    spdkPrefix= os.path.abspath(os.path.join(os.getcwd(), "../../../../")) + "/thirdparty/spdk"
    os.system('sudo ls')
    os.system('./unbindssd.sh')
    os.system('sudo '+ spdkPrefix+'/scripts/setup.sh')
    os.system('sudo '+ spdkPrefix+'/scripts/setup.sh status')
    # next, download corpus
    
    # gen10MDPR()


if __name__ == "__main__":
    main()
