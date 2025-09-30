cd ..

# create conda environment
env_name='para'
conda env update -n $env_name -f reqs/conda.yaml
conda activate $env_name

# install some dependencies
conda install nvidia/label/cuda-12.2.1::cuda-toolkit -y
conda install nvidia/label/cuda-12.2.1::libcublas -y
conda install conda-forge::cudnn -y
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

CPLUS_INCLUDE_PATH=/.../cuda/12.2.1/include:$CPLUS_INCLUDE_PATH BUILD_EXT=1 pip install colossalai
pip install -r reqs/pip_extra.txt

# install Apex
cd ..
export apex_commit=810ffae374a2b9cb4b5c5e28eaeca7d7998fca0c
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout $apex_commit
# may have to comment out version checking first in setup.py before installing
# while i install apex i also need to export TORCH_CUDA_ARCH_LIST="8.0" (for A100) and export CC=/gcc/11.2.0/bin/gcc; export CXX=/gcc/11.2.0/bin/g++; export CUDAHOSTCXX=/gcc/11.2.0/bin/g++
CPLUS_INCLUDE_PATH=/usr/local/cuda-12.2/include:$CPLUS_INCLUDE_PATH LD_LIBRARY_PATH=/usr/local/cuda-12.2/include:$LD_LIBRARY_PATH pip install . -v --no-build-isolation --disable-pip-version-check --no-cache-dir --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam --group_norm"
cd ../parallel-opt

pip install -e .
