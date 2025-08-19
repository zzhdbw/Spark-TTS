from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

model_dir = "/home/zzh/code/TTS/Spark-TTS/pretrained_models/Spark-TTS-0.5B"
dataset_path = "/home/zzh/code/TTS/Spark-TTS/data/yuanshen_format_dataset.parquet"
max_seq_length = 8192

##########################
dataset = Dataset.from_parquet(dataset_path)


# tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/LLM")
# model = AutoModelForCausalLM.from_pretrained(f"{model_dir}/LLM")

trainer = SFTTrainer(
    model=f"{model_dir}/LLM",
    train_dataset=dataset,
    # tokenizer=tokenizer,
    args=SFTConfig(
        deepspeed="config/ds_z2_config.json",
        max_length=max_seq_length,
        dataset_text_field="text",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        use_liger_kernel=True,
        packing=True,  # Can make training 5x faster for short sequences.
        warmup_steps=5,
        num_train_epochs=2,  # Set this for 1 full training run.
        learning_rate=2e-5,
        fp16=False,  # We're doing full float32 s disable mixed precision
        bf16=True,  # We're doing full float32 s disable mixed precision
        logging_steps=1,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="swanlab",  # Use this for WandB etc
    ),
)
trainer.train()
