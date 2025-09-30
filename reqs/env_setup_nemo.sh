cd ..

#https://docs.nvidia.com/nemo-framework/user-guide/latest/installation.html

conda create --name nemo python==3.10.12 -y
conda activate nemo

conda install pytorch==2.4.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
# conda install nvidia/label/cuda-12.2.1::cuda-toolkit -y
# conda install nvidia/label/cuda-12.2.1::libcublas -y
# conda install conda-forge::cudnn -y
# pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121


pip install Cython packaging
pip install nemo_toolkit['all']
conda install -c conda-forge mpi4py mpich -y


export apex_commit=810ffae374a2b9cb4b5c5e28eaeca7d7998fca0c
export te_commit=bfe21c3d68b0a9951e5716fb520045db53419c5e
# export mcore_commit=02871b4df8c69fac687ab6676c4246e936ce92d0
export mcore_commit=a407351af36545af03ebeecb6eed9353dd76b53e


# apex
conda install -c nvidia cuda-nvprof=11.8 -y
pip install packaging
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout $apex_commit
# may have to comment out version checking first in setup.py (around line 40) before installing
CPLUS_INCLUDE_PATH=/usr/local/cuda-12.2/include:$CPLUS_INCLUDE_PATH LD_LIBRARY_PATH=/usr/local/cuda-12.2/include:$LD_LIBRARY_PATH pip install . -v --no-build-isolation --disable-pip-version-check --no-cache-dir --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam --group_norm"
cd ..


# transformer engine
# git clone https://github.com/NVIDIA/TransformerEngine.git
# cd TransformerEngine
# git checkout $te_commit
# git submodule init && git submodule update
# conda install openmpi
# CUDACXX=/usr/local/cuda-12.2/bin/nvcc NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install .
# cd ..
# CUDACXX=/usr/local/cuda-12.2/bin/nvcc NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 pip install transformer_engine[pytorch]


# megatron core
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout $mcore_commit
pip install .
cd megatron/core/datasets
make
cd ../../../..


# nemo run
pip install git+https://github.com/NVIDIA/NeMo-Run.git


# other dependencies for NeMO to match versions
pip install --no-binary=opencc opencc==1.1.9 --force-reinstall
CPATH=/usr/local/cuda-12.2/targets/x86_64-linux/include:$CPATH LD_LIBRARY_PATH=/usr/local/cuda-12.2/targets/x86_64-linux/lib:$LD_LIBRARY_PATH PATH=/usr/local/cuda-12.2/bin:$PATH pip install transformer_engine[pytorch]==1.12.0
pip install flash-attn==2.6.3 --no-build-isolation


# install deepspeed so it can be used
pip install deepspeed==0.16.4
pip install peft==0.14.0


# install evaluation functions
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
cd ..


# install our package
cd joint-optim
pip install -r reqs/pip_extra.txt
pip install -e .