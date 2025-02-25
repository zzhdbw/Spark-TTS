#!/bin/bash

# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Get the absolute path of the script's directory
script_dir=$(dirname "$(realpath "$0")")

# Get the root directory
root_dir=$(dirname "$script_dir")

# Set default parameters
device=0
save_dir='example/results'
model_dir="pretrained_models/Spark-TTS-0.5B"
text="精准合成，自然呈现，为你开启语音合成新纪元。本模型由智能语音开源社区 Spark Audio 精心打造。"
prompt_text="吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。"
prompt_speech_path="example/prompt_audio.wav"

# Change directory to the root directory
cd "$root_dir" || exit

source utils/parse_options.sh

# Run inference for each JSON file
python inference.py \
    --text "${text}" \
    --device "${device}" \
    --save_dir "${save_dir}" \
    --model_dir "${model_dir}" \
    --prompt_text "${prompt_text}" \
    --prompt_speech_path "${prompt_speech_path}"
    
    