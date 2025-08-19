# python 1_raw_to_ds.py \
#     --folder_path "/home/zzh/code/Develop/ray_learn/yuanshen" \
#     --output_parquet "./temp/yuanshen_dataset.parquet"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
python -m data.2_encode_audio \
    --model_path "/home/zzh/code/TTS/Spark-TTS/pretrained_models/Spark-TTS-0.5B-unsloth/" \
    --input_parquet "data/temp/yuanshen_dataset.parquet" \
    --output_parquet "data/yuanshen_format_dataset.parquet" \
    --num_cpus 24 \
    --num_gpus 6 \
    --num_gpus_per_worker 0.8 \
    --num_workers 6 \
    --num_proc 8 

