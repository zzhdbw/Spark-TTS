# unsloth微调，节省显存
# export CUDA_VISIBLE_DEVICES=0
# nohup python -m finetune.unsloth_train > logs/train.log 2>&1 &

# unsloth 推理
python -m finetune.unsloth_infer

# trl 支持多卡训练以加速训练过程
# deepspeed --include localhost:5,7  finetune/trl_train.py

