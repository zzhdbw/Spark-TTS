<div align="center">
    <h1>
    Spark-TTS
    </h1>
    <p>
    Official PyTorch code for inference of <br>
    <b><em>Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens</em></b>
    </p>
    <p>
    <img src="src/logo.webp" alt="Spark-TTS Logo" style="width: 200px; height: 200px;">
    </p>
    <p>
    </p>
    <a href="https://spark-tts.github.io/"><img src="https://img.shields.io/badge/Demo-Page-lightgrey" alt="version"></a>
    <a href="https://github.com/SparkAudio/Spark-TTS"><img src="https://img.shields.io/badge/Platform-linux-lightgrey" alt="version"></a>
    <a href="https://github.com/SparkAudio/Spark-TTS"><img src="https://img.shields.io/badge/Python-3.12.2-orange" alt="version"></a>
    <a href="https://github.com/SparkAudio/Spark-TTS"><img src="https://img.shields.io/badge/PyTorch-2.5.1-brightgreen" alt="python"></a>
    <a href="https://github.com/SparkAudio/Spark-TTS"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="mit"></a>
</div>


## Spark-TTS 🔥

### Overview

Spark-TTS is an advanced text-to-speech system that uses the power of large language models (LLM) for highly accurate and natural-sounding voice synthesis. It is designed to be efficient, flexible, and powerful for both research and production use.

### Key Features

- **Simplicity and Efficiency**: Built entirely on Qwen2.5, Spark-TTS eliminates the need for additional generation models like flow matching. Instead of relying on separate models to generate acoustic features, it directly reconstructs audio from the code predicted by the LLM. This approach streamlines the process, improving efficiency and reducing complexity.
- **High-Quality Voice Cloning**: Supports zero-shot voice cloning, which means it can replicate a speaker's voice even without specific training data for that voice. This is ideal for cross-lingual and code-switching scenarios, allowing for seamless transitions between languages and voices without requiring separate training for each one.
- **Bilingual Support**: Supports both Chinese and English, and is capable of zero-shot voice cloning for cross-lingual and code-switching scenarios, enabling the model to synthesize speech in multiple languages with high naturalness and accuracy.

## Install
**Clone and Install**

- Clone the repo
``` sh
git clone https://github.com/SparkAudio/Spark-TTS.git
cd Spark-TTS
```

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:

``` sh
conda create -n sparktts -y python=3.12
conda activate sparktts
pip install -r requirements.txt
# if you are in mainland of China, you can set the mirror as follows:
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

**Model Download**

TBD

**Basic Usage**

You can simply run the demo with the following commands:
``` sh
cd example
bash infer.sh
```
