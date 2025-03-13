export N_GPUS=8
export BASE_MODEL=Qwen/Qwen2.5-0.5B
export DATA_DIR=./data/alfworld
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=alfworld-qwen2.5-0.5b
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh