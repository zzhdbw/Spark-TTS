export CUDA_VISIBLE_DEVICES=0
nohup python -m finetune.spark_tts_train > logs/train.log 2>&1 &