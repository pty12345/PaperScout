set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_V1=1
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_HOME=/usr/local/cuda
export HYDRA_FULL_ERROR=1
export MLFLOW_TRACKING_URI=http://127.0.0.1:8080


PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/recipe/paper_search/base.yaml"

PROJECT_NAME='FALCON'
EXP_NAME='falcon-v3-vp-lock'

actor_ppo_max_token_len_per_gpu=45056
critic_ppo_max_token_len_per_gpu=131072
# critic_ppo_max_token_len_per_gpu=45056

python3 -m arft.main_agent_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=$HOME/data/pasa/train.parquet \
    data.val_files=$HOME/data/pasa/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=10240 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/data/tingyue/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=320 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.agent.agent_flow_config_path=$CONFIG_PATH \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.rollout.trace.backend=mlflow \
    actor_rollout_ref.rollout.trace.token2text=True \
    actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker=5 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=/data/tingyue/workspace/oyj/AgentRFT-PaperSearch/checkpoints/Convert/BEST_v3_vp_step_100_lock_critic \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_mini_batch_size=128 \
    critic.ppo_micro_batch_size_per_gpu=8 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_ppo_max_token_len_per_gpu \
    critic.ppo_max_token_len_per_gpu=$critic_ppo_max_token_len_per_gpu \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab","mlflow"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=20 $@
