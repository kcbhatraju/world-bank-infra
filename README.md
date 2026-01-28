# How to Run
1. Download the xBD "Challenge train/test/hold" datasets from [here](https://github.com/fzmi/ubdd.git). Rename the files `train.tar`, `test.har`, and `hold.tar`.

2. SSH into login node: `ssh [sunet-id]@cme291-login.stanford.edu`

3. Create a conda environment for the project (and to transfer files). Even if you have one already, make a new one, otherwise dependencies will conflict. The last two lines are to create folders for our data.
```
conda create -n worldbank python=3.11
conda activate worldbank
pip install magic-wormhole
mkdir tars xbd
cd tars
```
4. On your local computer, install `magic-wormhole` using pip (as above) and run `wormhole send train.tar`. This will generate a code (something like `9-street-backpack`). Run `wormhole receive 9-street-backpack` on the login node to transfer the file. Do similarly for `test.tar` and `hold.tar`.

5. The `.tar` files should automatically be in the `tars` directory. Now run `tar -xvf train.tar test.tar hold.tar` to untar the files, and then the following to move the un-tared folders into a new directory.
```
mv train/ ../xbd/
mv test/ ../xbd/
mv hold/ ../
```

6. Clone our repo and move the data.
```
git clone https://github.com/kcbhatraju/world-bank-infra.git
mv xbd/ world-bank-infra/ubdd/
```

7. Activate GPU interactive terminal, and move into the correct folder and environment.
```
srun -p gpu-pascal --gres=gpu:1 --pty bash
cd world-bank-infra/ubdd/
conda activate worldbank
ulimit -n 65536
module load cuda/11.8
export CUDA_HOME=/opt/ohpc/pub/compiler/cuda/hpc_sdk/Linux_x86_64/23.7/cuda/11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPATH=/opt/ohpc/pub/compiler/cuda/hpc_sdk/Linux_x86_64/23.7/math_libs/11.8/include:$CPATH
export TORCH_CUDA_ARCH_LIST="6.0"
```
**You must run these every time you log into a GPU node.**

8. Install required packages (you will get an error when installing from `requirements.txt`, ignore it).
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt
pip install numpy opencv-python shapely
```

9. Hopefully, the following builds should now work.
```
cd models/dino/models/dino/ops
python setup.py build install
cd ~/world-bank-infra/GroundingDINO/
pip install -e . --no-build-isolation
cd ../ubdd/
```

10. Finally, preprocess and evaluate.
```
python3 datasets/preprocess-data.py -tsp xbd/train/ -vsp xbd/test/ -tssp xbd/hold/ -adp xbd/hold/
CUDA_VISIBLE_DEVICES=0 python predict-pretrain.py   --test-set-path "xbd/test/"   --dino-path "checkpoints/ubdd-dino-resnet.pth"   --dino-config "models/dino/config/DINO_4scale_UBDD_resnet.py"   --sam-path "checkpoints/sam_vit_h_4b8939.pth" --save-annotations

```

# Issue
The pre-processing seems to be fine -- for pre-disaster images, the output meshes correctly detect buildings. For post-disaster images, the damage is on a scale of 1-5, which we perceive as pitch black, but numerically, the class values are still present.

However, the outputs from `--save-annotations` in the evaluation step is completely, numerically zero across all images (verified with running `np.unique(out_image) = [0]`), which obviously should not be happening.
