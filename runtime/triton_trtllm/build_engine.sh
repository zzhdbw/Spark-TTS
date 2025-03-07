


# model_dir=./Qwen2.5-0.5B-Instruct/
# output_dir=./tllm_checkpoint_1gpu_fp16
# trt_engines_dir=./trt_engines

model_dir=/workspace_yuekai/spark-tts/Spark-TTS/pretrained_models/Spark-TTS-0.5B/LLM
base_name=Spark-TTS-0.5B
dtype=bfloat16
output_dir=./${base_name}_tllm_checkpoint_1gpu_${dtype}
trt_engines_dir=./${base_name}_trt_engines_1gpu_${dtype}


# python convert_checkpoint.py --model_dir $model_dir \
#                               --output_dir $output_dir \
#                               --dtype $dtype || exit 1

trtllm-build --checkpoint_dir $output_dir \
            --output_dir $trt_engines_dir \
            --max_batch_size 16 \
            --max_num_tokens 32768 \
            --gemm_plugin $dtype || exit 1
# trtllm-build --checkpoint_dir $output_dir \
#             --output_dir $trt_engines_dir \
#             --max_batch_size 16 \
#             --max_num_tokens 32768 \
#             --gemm_plugin $dtype || exit 1

python3 ../run.py --input_file  /workspace_yuekai/spark-tts/Spark-TTS/model_inputs.npy \
                  --max_output_len=1500 \
                  --tokenizer_dir $model_dir \
                  --top_k 50 \
                  --top_p 0.95 \
                  --temperature 0.8 \
                  --output_npy ./output.npy \
                  --engine_dir=$trt_engines_dir || exit 1


# python3 ../run.py --input_file  /workspace_yuekai/spark-tts/Spark-TTS/model_inputs.npy \
#                   --max_output_len=1500 \
#                   --tokenizer_dir $model_dir \
#                   --top_k 50 \
#                   --top_p 0.95 \
#                   --temperature 0.8 \
#                   --engine_dir=$trt_engines_dir || exit 1