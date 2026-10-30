# source this before running Lyra-2:  source third_party/lyra2_env.sh
cd ${H2W_REPO_ROOT}/third_party/lyra/Lyra-2
source .venv/bin/activate
SITE=$VIRTUAL_ENV/lib/python3.10/site-packages
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH="$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LYRA_DA3=${H2W_DATA_ROOT}/checkpoints/da3/DA3NESTED-GIANT-LARGE-1.1   # pass as --da3_model_name $LYRA_DA3
