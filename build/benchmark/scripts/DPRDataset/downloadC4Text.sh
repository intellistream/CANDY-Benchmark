sudo apt-get install git-lfs
rm -rf c4
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
cd c4
git lfs pull --include "c4-train.00000-of-01024.json.gz"
git lfs pull --include "c4-validation.00000-of-00008.json.gz"
git lfs pull --include "c4-train.00001-of-01024.json.gz"
touch c4done
