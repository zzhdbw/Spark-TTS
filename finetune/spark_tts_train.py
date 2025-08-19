# -*- coding: utf-8 -*-

from unsloth import FastModel
from huggingface_hub import snapshot_download
from datasets import load_dataset
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.audio import audio_volume_normalize
from trl import SFTConfig, SFTTrainer
import torchaudio.transforms as T
import torch
import sys
from datasets import load_from_disk

from datasets import Dataset

sys.path.append('Spark-TTS')

max_seq_length = 4096  # Choose any for long context!

# Download model and code
# snapshot_download(
#     "unsloth/Spark-TTS-0.5B",
#     local_dir="/home/zzh/code/TTS/Spark-TTS/pretrained_models/Spark-TTS-0.5B-unsloth",
# )


model, tokenizer = FastModel.from_pretrained(
    model_name=f"/home/zzh/code/TTS/Spark-TTS/pretrained_models/Spark-TTS-0.5B-unsloth/LLM",
    max_seq_length=max_seq_length,
    dtype=torch.float32,  # Spark seems to only work on float32 for now
    full_finetuning=True,  # We support full finetuning now!
    load_in_4bit=False,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

"""We now add LoRA adapters so we only need to update 1 to 10% of all parameters!"""

# LoRA does not work with float32 only works with bfloat16 !!!
model = FastModel.get_peft_model(
    model,
    r=128,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=128,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

"""<a name="Data"></a>
### Data Prep  

We will use the `MrDragonFox/Elise`, which is designed for training TTS models.
Ensure that your dataset follows the required format: **text, audio** for single-speaker models or **source, text, audio** for multi-speaker models. 
You can modify this section to accommodate your own dataset, but maintaining the correct structure is essential for optimal training.
"""


# dataset = load_dataset("MrDragonFox/Elise", split="train")
# dataset = load_from_disk("./data/dataset")


# @title Tokenization Function


audio_tokenizer = BiCodecTokenizer(
    "/home/zzh/code/TTS/Spark-TTS/pretrained_models/Spark-TTS-0.5B-unsloth", "cuda"
)


dataset = Dataset.from_parquet(
    "/home/zzh/code/TTS/Spark-TTS/data/yuanshen_format_dataset.parquet"
)

print("Moving Bicodec model and Wav2Vec2Model to cpu.")
audio_tokenizer.model.cpu()
audio_tokenizer.feature_extractor.cpu()
torch.cuda.empty_cache()

"""<a name="Train"></a>
### Train the model
Now let's train our model. 
We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.
 We also support TRL's `DPOTrainer`!
"""

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=False,  # Can make training 5x faster for short sequences.
    args=SFTConfig(
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        warmup_steps=5,
        num_train_epochs=2,  # Set this for 1 full training run.
        learning_rate=2e-4,
        fp16=False,  # We're doing full float32 s disable mixed precision
        bf16=False,  # We're doing full float32 s disable mixed precision
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="swanlab",  # Use this for WandB etc
    ),
)

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
