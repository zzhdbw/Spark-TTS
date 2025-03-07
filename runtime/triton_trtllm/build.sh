
pip install -r /workspace_yuekai/spark-tts/Spark-TTS/requirements.txt
model_repo=./model_repo_test
rm -rf $model_repo

cp -r ./model_repo $model_repo

ENGINE_PATH=/workspace_yuekai/spark-tts/TensorRT-LLM/examples/qwen/Spark-TTS-0.5B_trt_engines_1gpu_bfloat16
MAX_QUEUE_DELAY_MICROSECONDS=0
gpu_device_ids=0
python3 fill_template.py -i ${model_repo}/tensorrt_llm/config.pbtxt gpu_device_ids:${gpu_device_ids},triton_backend:tensorrtllm,triton_max_batch_size:16,decoupled_mode:False,max_beam_width:1,engine_dir:${ENGINE_PATH},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS},encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32

# enable_context_fmha_fp32_acc:${ENABLE_CONTEXT_FMHA_FP32_ACC}
export PYTHONPATH=/workspace_yuekai/spark-tts/Spark-TTS/
CUDA_VISIBLE_DEVICES=${gpu_device_ids} tritonserver --model-repository ${model_repo}