from unsloth import FastModel
from huggingface_hub import snapshot_download
from datasets import load_dataset
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.audio import audio_volume_normalize
from trl import SFTConfig, SFTTrainer
import re
import numpy as np
from typing import Dict, Any
import torchaudio.transforms as T
import locale
import os
import torch

import numpy as np
import sys

sys.path.append('Spark-TTS')

"""<a name="Inference"></a>
### Inference
Let's run the model! You can change the prompts

"""

input_text = "改革春风吹满地, 中国人民真争气！"
chosen_voice = "雪艳"  # None for single-speaker

# @title Run Inference

model, tokenizer = FastModel.from_pretrained(
    model_name=f"/home/zzh/code/TTS/Spark-TTS/outputs/checkpoint-316",
    max_seq_length=2048,
    dtype=torch.float32,  # Spark seems to only work on float32 for now
    full_finetuning=True,  # We support full finetuning now!
    load_in_4bit=False,
)
FastModel.for_inference(model)  # Enable native 2x faster inference
audio_tokenizer = BiCodecTokenizer(
    "/home/zzh/code/TTS/Spark-TTS/pretrained_models/Spark-TTS-0.5B-unsloth", "cuda"
)


@torch.inference_mode()
def generate_speech_from_text(
    text: str,
    temperature: float = 0.8,  # Generation temperature
    top_k: int = 50,  # Generation top_k
    top_p: float = 1,  # Generation top_p
    max_new_audio_tokens: int = 2048,  # Max tokens for audio part
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> np.ndarray:
    """
    Generates speech audio from text using default voice control parameters.

    Args:
        text (str): The text input to be converted to speech.
        temperature (float): Sampling temperature for generation.
        top_k (int): Top-k sampling parameter.
        top_p (float): Top-p (nucleus) sampling parameter.
        max_new_audio_tokens (int): Max number of new tokens to generate (limits audio length).
        device (torch.device): Device to run inference on.

    Returns:
        np.ndarray: Generated waveform as a NumPy array.
    """

    torch.compiler.reset()

    prompt = "".join(
        [
            "<|task_tts|>",
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_global_token|>",
        ]
    )

    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

    print("Generating token sequence...")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_audio_tokens,  # Limit generation length
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,  # Stop token
        pad_token_id=tokenizer.pad_token_id,  # Use models pad token id
    )
    print("Token sequence generated.")

    generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1] :]

    predicts_text = tokenizer.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False
    )[0]
    # print(f"\nGenerated Text (for parsing):\n{predicts_text}\n") # Debugging

    # Extract semantic token IDs using regex
    semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", predicts_text)
    if not semantic_matches:
        print("Warning: No semantic tokens found in the generated output.")
        # Handle appropriately - perhaps return silence or raise error
        return np.array([], dtype=np.float32)

    pred_semantic_ids = (
        torch.tensor([int(token) for token in semantic_matches]).long().unsqueeze(0)
    )  # Add batch dim

    # Extract global token IDs using regex (assuming controllable mode also generates these)
    global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", predicts_text)
    if not global_matches:
        print(
            "Warning: No global tokens found in the generated output (controllable mode). Might use defaults or fail."
        )
        pred_global_ids = torch.zeros((1, 1), dtype=torch.long)
    else:
        pred_global_ids = (
            torch.tensor([int(token) for token in global_matches]).long().unsqueeze(0)
        )  # Add batch dim

    pred_global_ids = pred_global_ids.unsqueeze(0)  # Shape becomes (1, 1, N_global)

    print(f"Found {pred_semantic_ids.shape[1]} semantic tokens.")
    print(f"Found {pred_global_ids.shape[2]} global tokens.")

    # 5. Detokenize using BiCodecTokenizer
    print("Detokenizing audio tokens...")
    # Ensure audio_tokenizer and its internal model are on the correct device
    audio_tokenizer.device = device
    audio_tokenizer.model.to(device)
    # Squeeze the extra dimension from global tokens as seen in SparkTTS example
    wav_np = audio_tokenizer.detokenize(
        pred_global_ids.to(device).squeeze(0),  # Shape (1, N_global)
        pred_semantic_ids.to(device),  # Shape (1, N_semantic)
    )
    print("Detokenization complete.")

    return wav_np


if __name__ == "__main__":
    print(f"Generating speech for: '{input_text}'")
    text = f"{chosen_voice}: " + input_text if chosen_voice else input_text
    generated_waveform = generate_speech_from_text(input_text)

    if generated_waveform.size > 0:
        import soundfile as sf

        output_filename = "outputs/generated_speech_controllable.wav"
        sample_rate = audio_tokenizer.config.get("sample_rate", 16000)
        sf.write(output_filename, generated_waveform, sample_rate)
        print(f"Audio saved to {output_filename}")

        # # Optional: Play in notebook
        # from IPython.display import Audio, display

        # display(Audio(generated_waveform, rate=sample_rate))
        # 保存到本地
        output_path = "outputs/generated_speech_controllable.wav"
        sf.write(output_path, generated_waveform, sample_rate)
    else:
        print("Audio generation failed (no tokens found?).")


def sava_model():
    """<a name="Save"></a>
    ### Saving, loading finetuned models
    To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.

    **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!
    """

    # model.save_pretrained("lora_model")  # Local saving
    # tokenizer.save_pretrained("lora_model")
    # model.push_to_hub("your_name/lora_model", token = "...") # Online saving
    # tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving

    """### Saving to float16

    We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.
    """

    # Merge to 16bit
    if False:
        model.save_pretrained_merged(
            "model",
            tokenizer,
            save_method="merged_16bit",
        )
    if False:
        model.push_to_hub_merged(
            "hf/model", tokenizer, save_method="merged_16bit", token=""
        )

    # Merge to 4bit
    if False:
        model.save_pretrained_merged(
            "model",
            tokenizer,
            save_method="merged_4bit",
        )
    if False:
        model.push_to_hub_merged(
            "hf/model", tokenizer, save_method="merged_4bit", token=""
        )

    # Just LoRA adapters
    if False:
        model.save_pretrained("model")
        tokenizer.save_pretrained("model")
    if False:
        model.push_to_hub("hf/model", token="")
        tokenizer.push_to_hub("hf/model", token="")
