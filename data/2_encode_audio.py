import torch
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.audio import audio_volume_normalize
import librosa
import ray
from tqdm import tqdm
from light_util import get_time
from datasets import Dataset
import sys

sys.path.append('/home/zzh/code/TTS/Spark-TTS')


class BiCodecWrapper:
    def __init__(
        self,
        model_path="/home/zzh/code/TTS/Spark-TTS/pretrained_models/Spark-TTS-0.5B-unsloth",
        # target_sr=16000,
    ) -> None:
        self.audio_tokenizer = BiCodecTokenizer(
            model_path,
            "cuda",
        )
        # self.target_sr = target_sr

    @torch.inference_mode()
    def extract_wav2vec2_features(self, wavs: torch.Tensor) -> torch.Tensor:
        """extract wav2vec2 features"""

        if wavs.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, but got shape {wavs.shape}")
        wav_np = wavs.squeeze(0).cpu().numpy()

        processed = self.audio_tokenizer.processor(
            wav_np,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_values = processed.input_values

        input_values = input_values.to(self.audio_tokenizer.feature_extractor.device)

        model_output = self.audio_tokenizer.feature_extractor(
            input_values,
        )

        if model_output.hidden_states is None:
            raise ValueError(
                "Wav2Vec2Model did not return hidden states. Ensure config `output_hidden_states=True`."
            )

        num_layers = len(model_output.hidden_states)
        required_layers = [11, 14, 16]
        if any(l >= num_layers for l in required_layers):
            raise IndexError(
                f"Requested hidden state indices {required_layers} out of range for model with {num_layers} layers."
            )

        feats_mix = (
            model_output.hidden_states[11]
            + model_output.hidden_states[14]
            + model_output.hidden_states[16]
        ) / 3

        return feats_mix

    @torch.inference_mode()
    def formatting_audio_func(self, example):
        text = (
            f"{example['source']}: {example['text']}"
            if "source" in example
            else example["text"]
        )
        audio_array, sampling_rate = librosa.load(
            example["audio_path"], sr=self.audio_tokenizer.config['sample_rate']
        )
        # audio_array = example["audio"]["array"]
        # sampling_rate = example["audio"]["sampling_rate"]

        # target_sr = self.audio_tokenizer.config['sample_rate']

        # if sampling_rate != target_sr:
        #     resampler = T.Resample(orig_freq=sampling_rate, new_freq=target_sr)
        #     audio_tensor_temp = torch.from_numpy(audio_array).float()
        #     audio_array = resampler(audio_tensor_temp).numpy()

        if self.audio_tokenizer.config["volume_normalize"]:
            audio_array = audio_volume_normalize(audio_array)

        ref_wav_np = self.audio_tokenizer.get_ref_clip(audio_array)

        audio_tensor = (
            torch.from_numpy(audio_array)
            .unsqueeze(0)
            .float()
            .to(self.audio_tokenizer.device)
        )
        ref_wav_tensor = (
            torch.from_numpy(ref_wav_np)
            .unsqueeze(0)
            .float()
            .to(self.audio_tokenizer.device)
        )

        feat = self.extract_wav2vec2_features(audio_tensor)

        batch = {
            "wav": audio_tensor,
            "ref_wav": ref_wav_tensor,
            "feat": feat.to(self.audio_tokenizer.device),
        }

        semantic_token_ids, global_token_ids = self.audio_tokenizer.model.tokenize(
            batch
        )

        global_tokens = "".join(
            [
                f"<|bicodec_global_{i}|>"
                for i in global_token_ids.squeeze().cpu().numpy()
            ]  # Squeeze batch dim
        )
        semantic_tokens = "".join(
            [
                f"<|bicodec_semantic_{i}|>"
                for i in semantic_token_ids.squeeze().cpu().numpy()
            ]  # Squeeze batch dim
        )

        inputs = [
            "<|task_tts|>",
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_global_token|>",
            global_tokens,
            "<|end_global_token|>",
            "<|start_semantic_token|>",
            semantic_tokens,
            "<|end_semantic_token|>",
            "<|im_end|>",
        ]

        inputs = "".join(inputs)
        return {"text": inputs}


@get_time
def main():
    import argparse

    parse = argparse.ArgumentParser()
    parse.add_argument('--model_path', type=str, help='BiCodec model path')
    parse.add_argument('--input_parquet', type=str, help='input parquet file path')
    parse.add_argument('--output_parquet', type=str, help='onput parquet file path')
    parse.add_argument('--num_cpus', type=int, help='ray num_cpus')
    parse.add_argument('--num_gpus', type=int, help='ray num_gpus')
    parse.add_argument(
        '--num_gpus_per_worker', type=float, help='ray num_gpus_per_worker'
    )
    parse.add_argument('--num_workers', type=int, help='ray num_workers')
    parse.add_argument('--num_proc', type=int, help='dataset map num_proc')

    opt = parse.parse_args()

    ray.init(num_gpus=opt.num_gpus, num_cpus=opt.num_cpus)

    BiCodecWrapper_remote = ray.remote(num_gpus=opt.num_gpus_per_worker, num_cpus=2)(
        BiCodecWrapper
    )

    # 加载数据集
    print("Loading dataset...")
    ds = Dataset.from_parquet(opt.input_parquet)
    print(f"Dataset loaded with {len(ds)} samples")

    # 创建 BiCodec workers
    print("Creating BiCodec workers...")
    workers = []
    for gpu_id in range(opt.num_workers):
        worker = BiCodecWrapper_remote.remote(opt.model_path)
        workers.append(worker)
    print("Create BiCodec workers done")

    futures = []
    for index, d in enumerate(ds):
        future = workers[index % opt.num_workers].formatting_audio_func.remote(d)
        futures.append(future)

    all_results = []
    for future in tqdm(futures, desc="Processing audio"):
        all_results.append(ray.get(future))
    ray.shutdown()

    result_ds = Dataset.from_list(all_results)
    # result_ds = result_ds.remove_columns(["audio_path", "source"])

    print(len(all_results))
    print(f"Processing completed! Total processed: {len(all_results)} samples")
    print(result_ds[0])
    result_ds.to_parquet(opt.output_parquet)


if __name__ == "__main__":
    main()
