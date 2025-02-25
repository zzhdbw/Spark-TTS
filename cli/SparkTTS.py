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

import re
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.file import load_config
from models.audio_tokenizer import BiCodecTokenizer
from utils.token_parser import TASK_TOKEN_MAP


class SparkTTS:
    """
    Spark-TTS for text-to-speech generation.
    """

    def __init__(self, model_dir: Path, device: torch.device = torch.device("cuda:0")):
        """
        Initializes the SparkTTS model with the provided configurations and device.

        Args:
            model_dir (Path): Directory containing the model and config files.
            device (torch.device): The device (CPU/GPU) to run the model on.
        """
        self.device = device
        self.model_dir = model_dir
        self.configs = load_config(f"{model_dir}/config.yaml")
        self.sample_rate = self.configs["sample_rate"]
        self._initialize_inference()

    def _initialize_inference(self):
        """Initializes the tokenizer, model, and audio tokenizer for inference."""
        self.tokenizer = AutoTokenizer.from_pretrained(f"{self.model_dir}/LLM")
        self.model = AutoModelForCausalLM.from_pretrained(f"{self.model_dir}/LLM")
        self.audio_tokenizer = BiCodecTokenizer(self.model_dir, device=self.device)
        self.model.to(self.device)

    @torch.no_grad()
    def inference(
        self,
        text: str,
        prompt_speech_path: Path,
        prompt_text: str = None,
        temperature: float = 0.8,
        top_k: float = 50,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """
        Performs inference to generate speech from text, incorporating prompt audio and/or text.

        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path (Path): Path to the audio file used as a prompt.
            prompt_text (str, optional): Transcript of the prompt audio.
            temperature (float, optional): Sampling temperature for controlling randomness. Default is 0.8.
            top_k (float, optional): Top-k sampling parameter. Default is 50.
            top_p (float, optional): Top-p (nucleus) sampling parameter. Default is 0.95.

        Returns:
            torch.Tensor: Generated waveform as a tensor.
        """
        global_token_ids, semantic_token_ids = self.audio_tokenizer.tokenize(prompt_speech_path)
        global_tokens = "".join([f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()])

        # Prepare the input tokens for the model
        if prompt_text is not None:
            semantic_tokens = "".join([f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()])
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                prompt_text,
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
                "<|start_semantic_token|>",
                semantic_tokens,
            ]
        else:
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
            ]

        inputs = "".join(inputs)
        model_inputs = self.tokenizer([inputs], return_tensors="pt").to(self.device)

        # Generate speech using the model
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=3000,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

        # Trim the output tokens to remove the input tokens
        generated_ids = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # Decode the generated tokens into text
        predicts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Extract semantic token IDs from the generated text
        pred_semantic_ids = torch.tensor([int(token) for token in re.findall(r"\d+", predicts)]).long().unsqueeze(0)

        # Convert semantic tokens back to waveform
        wav = self.audio_tokenizer.detokenize(
            global_token_ids.to(self.device).squeeze(0),
            pred_semantic_ids.to(self.device),
        )

        return wav
