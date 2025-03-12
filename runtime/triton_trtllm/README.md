## Nvidia Triton Inference Serving Best Practice for Spark TTS

### Quick Start
Directly launch the service using docker compose.
```sh
docker compose up
```

### Build Image
Build the docker image from scratch. 
```sh
docker build . -f Dockerfile.server -t soar97/triton-spark-tts:25.02
```

### Create Docker Container
```sh
your_mount_dir=/mnt:/mnt
docker run -it --name "spark-tts-server" --gpus all --net host -v $your_mount_dir --shm-size=2g soar97/triton-spark-tts:25.02
```

### Export Models to TensorRT-LLM and Launch Server
Inside docker container, we would follow the official guide of TensorRT-LLM to build TensorRT-LLM engines. See [here](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/qwen).

```sh
bash run.sh 0 3
```
### Simple HTTP client
```sh
python3 client_http.py
```

### Benchmark using Dataset
```sh
num_task=2
python3 client_grpc.py --num-tasks $num_task --huggingface-dataset yuekai/seed_tts --split-name wenetspeech4tts
```

### Benchmark Results
Decoding on a single L20 GPU, using 26 different prompt_audio/target_text pairs.

| Model | Note   | Concurrency | Avg Latency     | RTF | 
|-------|-----------|-----------------------|---------|--|
| Spark-TTS-0.5B | [Code Commit]() | 4                   | 253 ms | 0.0394|