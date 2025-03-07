# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import json
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.dlpack import from_dlpack, to_dlpack

import triton_python_backend_utils as pb_utils

import math
import os
from functools import wraps

from transformers import AutoTokenizer

import numpy as np
import re
from typing import Tuple

from sparktts.utils.token_parser import LEVELS_MAP, GENDER_MAP, TASK_TOKEN_MAP


def process_prompt(
    text: str,
    prompt_text: str = None,
    global_token_ids: torch.Tensor = None,
    semantic_token_ids: torch.Tensor = None,
) -> Tuple[str, torch.Tensor]:
    """
    Process input for voice cloning.

    Args:
        text (str): The text input to be converted to speech.
        prompt_speech_path (Path): Path to the audio file used as a prompt.
        prompt_text (str, optional): Transcript of the prompt audio.

    Return:
        Tuple[str, torch.Tensor]: Input prompt; global tokens
    """

    # global_token_ids, semantic_token_ids = self.audio_tokenizer.tokenize(
    #     prompt_speech_path
    # )
    global_tokens = "".join(
        [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
    )
    print(global_tokens, 233333333333, len(global_tokens), "global_tokens")
    # Prepare the input tokens for the model
    if prompt_text is not None:
        semantic_tokens = "".join(
            [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()]
        )
        print(semantic_tokens, 233333333333, len(semantic_tokens), "semantic_tokens")
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

    return inputs, global_token_ids

class TritonPythonModel:
    def initialize(self, args):
        parameters = json.loads(args['model_config'])['parameters']
        for key, value in parameters.items():
            parameters[key] = value["string_value"]
        model_dir = parameters["model_dir"]
        self.tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/LLM")
        self.device = torch.device("cuda")
        self.decoupled = False

    def forward_llm(self, input_ids):
        """
        Prepares the response from the language model based on the provided
        inputs. Creates a `pb_utils.InferenceRequest` object with passed
        `llm_request_inputs` to send to a decoupled TensorRTLLM model.
        For each response from the language model:
            - Checks for errors and raise an exception if any are found.
            - Extracts the "output_ids" tensor from the response.
            - Determines the finish reason based on the presence of the
              end-of-sequence token or reaching the maximum length.
            - Appends the generated token IDs to `output_ids`.
            - If the finish reason is determined, decodes the output IDs to text
              and prepares the final response.

        The final response includes the generated text, finish reason,
        completion tokens, prompt tokens, and total tokens.

        Parameters
        ----------
        - llm_request_inputs (dict): A dictionary containing the inputs for the language model.

        Returns
        -------
        - pb_utils.InferenceResponse: The response object containing the generated text and additional metadata.
        """
        # convert input_ids to numpy, with shape [1, sequence_length]
        input_ids = input_ids.cpu().numpy()
        print(input_ids.shape, 233333333333, "input_ids")
        max_tokens = 512
        input_dict = {
            "request_output_len": np.array([[max_tokens]], dtype=np.int32),
            "end_id": np.array([[self.tokenizer.eos_token_id]], dtype=np.int32),
            "pad_id": np.array([[self.tokenizer.pad_token_id]], dtype=np.int32),
            "streaming": np.array([[self.decoupled]], dtype=np.bool_),
            "runtime_top_p": np.array([[0.95]], dtype=np.float32),
            "runtime_top_k": np.array([[50]], dtype=np.int32),
            "temperature": np.array([[0.8]], dtype=np.float32),
            "input_ids": input_ids,
            "input_lengths": np.array([[input_ids.shape[1]]], dtype=np.int32),
        }
        for k, v in input_dict.items():
            print(k, v.shape, 233333333333, v.dtype)
        # exit()
        input_tensor_list = [
            pb_utils.Tensor(k, v) for k, v in input_dict.items()
        ]
        # input_tensor_list.append(pb_utils.Tensor.from_dlpack(
        #     "input_ids", to_dlpack(input_ids)
        # ))
        llm_request = pb_utils.InferenceRequest(
            model_name="tensorrt_llm",
            requested_output_names=["output_ids", "sequence_length"],
            inputs=input_tensor_list,
        )
        print("=======================================")
        llm_response = llm_request.exec(decoupled=self.decoupled)
        if llm_response.has_error():
            raise pb_utils.TritonModelException(
                llm_response.error().message())
        output_ids = pb_utils.get_output_tensor_by_name(
            llm_response, "output_ids").as_numpy()
        seq_lens = pb_utils.get_output_tensor_by_name(
            llm_response, "sequence_length").as_numpy()
        print(seq_lens, 233333333333, "seq_lens")
        actual_output_ids = output_ids[0][0]
        actual_output_ids = actual_output_ids[:seq_lens[0][0]]
        print(actual_output_ids, 233333333333, "actual_output_ids")
        return actual_output_ids

    def forward_audio_tokenizer(self, wav, wav_len):
        # input_tensor_0 = pb_utils.Tensor.
        # input_tensor_1 = pb_utils.Tensor.from_dlpack("wav_len", to_dlpack(wav_len))
    
        inference_request = pb_utils.InferenceRequest(
            model_name='audio_tokenizer',
            requested_output_names=['global_tokens', 'semantic_tokens'],
            inputs=[wav, wav_len]
        )
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        else:
            global_tokens = pb_utils.get_output_tensor_by_name(inference_response,
                                                            'global_tokens')
            global_tokens = torch.utils.dlpack.from_dlpack(global_tokens.to_dlpack()).cpu()
            semantic_tokens = pb_utils.get_output_tensor_by_name(inference_response,
                                                            'semantic_tokens')
            semantic_tokens = torch.utils.dlpack.from_dlpack(semantic_tokens.to_dlpack()).cpu()
            return global_tokens, semantic_tokens

    def forward_vocoder(self, global_token_ids, pred_semantic_ids):
        global_token_ids = pb_utils.Tensor.from_dlpack("global_tokens", to_dlpack(global_token_ids))
        pred_semantic_ids = pb_utils.Tensor.from_dlpack("semantic_tokens", to_dlpack(pred_semantic_ids))
        inference_request = pb_utils.InferenceRequest(
            model_name='vocoder',
            requested_output_names=['waveform'],
            inputs=[global_token_ids, pred_semantic_ids]
        )
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        else:
            waveform = pb_utils.get_output_tensor_by_name(inference_response,
                                                        'waveform')
            waveform = torch.utils.dlpack.from_dlpack(waveform.to_dlpack()).cpu()
            return waveform
        
    def execute(self, requests):
        # reference_text_list, target_text_list, reference_wav_list, reference_wav_ref_clip_list = [], [], [], []
        responses = []
        for request in requests:
            wav = pb_utils.get_input_tensor_by_name(request, "reference_wav")
            wav_len = pb_utils.get_input_tensor_by_name(
                request, "reference_wav_len")
            global_tokens, semantic_tokens = self.forward_audio_tokenizer(wav, wav_len)
            # print(wav_tensor.shape, wav_len.shape, 233333333333)
            # reference_wav_list.append(wav)
            # wav_ref_clip = self.get_ref_clip(wav[:, :wav_len])
            # reference_wav_ref_clip_list.append(wav_ref_clip)


            reference_text = pb_utils.get_input_tensor_by_name(
                request, "reference_text").as_numpy()
            reference_text = reference_text[0][0].decode('utf-8')
            # reference_text_list.append(reference_text)

            target_text = pb_utils.get_input_tensor_by_name(
                request, "target_text").as_numpy()
            target_text = target_text[0][0].decode('utf-8')
            # target_text_list.append(target_text)
            
            # ref_wav_clip_tensor = torch.cat(reference_wav_ref_clip_list, dim=0)
            # wav2vec2_features = self.model.audio_tokenizer.extract_wav2vec2_features(reference_wav_list)
            # audio_tokenizer_input_dict = {
            #     "ref_wav": ref_wav_clip_tensor, # no padding, spaker encoder
            #     "feat": wav2vec2_features,
            # }
            
            prompt, global_token_ids = process_prompt(
                text=target_text,
                prompt_text=reference_text,
                global_token_ids=global_tokens,
                semantic_token_ids=semantic_tokens,
            )
            print(semantic_tokens.shape, "semantic_tokens")
            print(global_tokens.shape, "global_tokens")
            print(prompt, "prompt", len(prompt))
            model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
            print(model_inputs, "model_inputs")
            input_ids = model_inputs.input_ids.to(torch.int32)
            print(input_ids.shape, 233333333333, 455555555)

            generated_ids = self.forward_llm(input_ids)
            print(generated_ids, "generated_ids", len(generated_ids))
            predicts = self.tokenizer.batch_decode([generated_ids], skip_special_tokens=True)[0]
            print(predicts, "predicts", len(predicts))
            pred_semantic_ids = (
                torch.tensor([int(token) for token in re.findall(r"bicodec_semantic_(\d+)", predicts)])
                .unsqueeze(0).to(torch.int32)
            )
            print(global_token_ids.shape, "global_token_ids")
            print(pred_semantic_ids.shape, "pred_semantic_ids")
            audio = self.forward_vocoder(
                global_token_ids.to(self.device),
                pred_semantic_ids.to(self.device),
            )

            audio = pb_utils.Tensor.from_dlpack("waveform", to_dlpack(audio))
            inference_response = pb_utils.InferenceResponse(output_tensors=[audio])
            responses.append(inference_response)
                             
        return responses
