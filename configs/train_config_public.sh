#!/bin/bash
# ============================================================
# Qwen3-8B 古诗多音字平仄消歧 GRPO 训练配置（公开版本）
#
# 注意：此配置已删除所有敏感的 API Key、绝对路径和内部信息。
# 请在使用前设置环境变量：
#   export WANDB_API_KEY="${WANDB_API_KEY:?Please set WANDB_API_KEY}"
#   export MODEL_PATH="${MODEL_PATH:?Please set MODEL_PATH}"
#   export DATASET_ROOT_DIR="${DATASET_ROOT_DIR:?Please set DATASET_ROOT_DIR}"
# ============================================================
set -x

# --- 必须由用户设置的环境变量 ---
export WANDB_API_KEY="${WANDB_API_KEY:?Please set WANDB_API_KEY}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"

# --- Ray 分布式配置 ---
ray stop --force
ray start --head --num-gpus="${NUM_GPUS:-3}" --port=6379
export RAY_DEBUG=legacy

# --- 路径配置 ---
CKPT_DIR="${CKPT_DIR:-./output}"
DATASET_ROOT_DIR="${DATASET_ROOT_DIR:?Please set DATASET_ROOT_DIR}"
MODEL_PATH="${MODEL_PATH:?Please set MODEL_PATH}"

# --- WandB 项目配置 ---
PROJECT_NAME="${PROJECT_NAME:-rft_duoyinzi}"
EXP_NAME="${EXP_NAME:-qwen3_8b_grpo}"
FORMATTED_TIME=$(date +"%Y%m%d%H%M%S")
FULL_NAME="${EXP_NAME}_${FORMATTED_TIME}"
LOG_DIR="${LOG_DIR:-./logs}"

# --- GRPO 训练主命令 ---
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATASET_ROOT_DIR}/train.parquet" \
    data.val_files="${DATASET_ROOT_DIR}/valid.parquet" \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-504}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-1500}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-2048}" \
    +data.data_source="duoyinzi" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr="${LR:-1e-6}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH:-252}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH:-12}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=22000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_COEF:-0.0005}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.n="${ROLLOUT_N:-4}" \
    actor_rollout_ref.rollout.temperature="${TEMPERATURE:-0.8}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL:-0.7}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.model.trust_remote_code=True \
    algorithm.kl_ctrl.kl_coef="${KL_COEF:-0.0005}" \
    trainer.critic_warmup=5 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${FULL_NAME}" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node="${NUM_GPUS:-3}" \
    actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS:-4}" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ:-30}" \
    trainer.test_freq="${TEST_FREQ:-30}" \
    trainer.log_val_generations=2 \
    trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
    trainer.total_training_steps="${TOTAL_STEPS:-null}" \
    trainer.balance_batch=True \
    trainer.resume_mode=auto \
    reward_model.reward_manager=prime \
    trainer.default_local_dir="${CKPT_DIR}/${FULL_NAME}" \
    2>&1 | tee "${LOG_DIR}/${FULL_NAME}.log"
