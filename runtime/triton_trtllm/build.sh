
# pip install -r /workspace_yuekai/spark-tts/Spark-TTS/requirements.txt
export PYTHONPATH=/workspace_yuekai/spark-tts/Spark-TTS/

model_repo=./model_repo_test
rm -rf $model_repo
cp -r ./model_repo $model_repo

ENGINE_PATH=/workspace_yuekai/spark-tts/TensorRT-LLM/examples/qwen/Spark-TTS-0.5B_trt_engines_1gpu_bfloat16
MAX_QUEUE_DELAY_MICROSECONDS=0
MODEL_DIR=/workspace_yuekai/spark-tts/Spark-TTS/pretrained_models/Spark-TTS-0.5B
LLM_TOKENIZER_DIR=/workspace_yuekai/spark-tts/Spark-TTS/pretrained_models/Spark-TTS-0.5B/LLM
BLS_INSTANCE_NUM=4
TRITON_MAX_BATCH_SIZE=16

python3 scripts/fill_template.py -i ${model_repo}/vocoder/config.pbtxt model_dir:${MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
python3 scripts/fill_template.py -i ${model_repo}/audio_tokenizer/config.pbtxt model_dir:${MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
python3 scripts/fill_template.py -i ${model_repo}/spark_tts/config.pbtxt bls_instance_num:${BLS_INSTANCE_NUM},llm_tokenizer_dir:${LLM_TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
python3 scripts/fill_template.py -i ${model_repo}/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:False,max_beam_width:1,engine_dir:${ENGINE_PATH},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS},encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32


CUDA_VISIBLE_DEVICES=0 tritonserver --model-repository ${model_repo}









