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

from sparktts.models.bicodec import BiCodec

class TritonPythonModel:
    def initialize(self, args):
        parameters = json.loads(args['model_config'])['parameters']
        for key, value in parameters.items():
            parameters[key] = value["string_value"]
        model_dir = parameters["model_dir"]
        self.device = torch.device("cuda")
        self.vocoder = BiCodec.load_from_checkpoint(f"{model_dir}/BiCodec").to(
            self.device
        )

    def execute(self, requests):
        global_tokens_list, semantic_tokens_list = [], []

        for request in requests:
            global_tokens_tensor = pb_utils.get_input_tensor_by_name(request, "global_tokens").as_numpy()
            semantic_tokens_tensor = pb_utils.get_input_tensor_by_name(request, "semantic_tokens").as_numpy()
            # check shape
            global_tokens_list.append(torch.from_numpy(global_tokens_tensor).to(self.device))
            semantic_tokens_list.append(torch.from_numpy(semantic_tokens_tensor).to(self.device))

        global_tokens = torch.cat(global_tokens_list, dim=0)
        semantic_tokens = torch.cat(semantic_tokens_list, dim=0)
        print(global_tokens.shape, semantic_tokens.shape, 233333333333, "global_tokens, semantic_tokens")

        wavs = self.vocoder.detokenize(semantic_tokens, global_tokens.unsqueeze(1))

        responses = []
        for i in range(len(requests)):
            wav_tensor = pb_utils.Tensor.from_dlpack("waveform", to_dlpack(wavs[i]))
            inference_response = pb_utils.InferenceResponse(output_tensors=[wav_tensor])
            responses.append(inference_response)
                             
        return responses
