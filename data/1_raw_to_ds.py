import os
from datasets import Dataset, Audio


def read_lab(example):
    try:
        example["text"] = (
            open(example["text_path"], "r", encoding="utf8").read().strip()
        )
    except Exception as e:
        print(f"Error reading lab file {example['text_path']}: {e}")
        example["text"] = ""
    return example


if __name__ == "__main__":
    import argparse

    parse = argparse.ArgumentParser()
    parse.add_argument('--folder_path', type=str, help='dataset folder path')
    parse.add_argument('--output_parquet', type=str, help='output parquet file path')

    opt = parse.parse_args()

    ##############################################
    data_list = []
    name_folder_list = [
        os.path.join(opt.folder_path, name)
        for name in os.listdir(opt.folder_path)
        if os.path.isdir(os.path.join(opt.folder_path, name))
    ]
    print(len(name_folder_list))

    for name_folder in name_folder_list:
        audio_paths = [
            audio_name.replace(".wav", "")
            for audio_name in os.listdir(name_folder)
            if audio_name.endswith('.wav')
        ]
        for audio_name in audio_paths:
            data_list.append(
                {
                    "source": name_folder.split("/")[-1],
                    "audio_path": os.path.join(
                        name_folder,
                        audio_name + ".wav",
                    ),
                    "text_path": os.path.join(
                        name_folder,
                        audio_name + ".lab",
                    ),
                }
            )
    ds = Dataset.from_list(data_list)

    ds = ds.map(read_lab, remove_columns=["text_path"], num_proc=4)
    ds = ds.filter(lambda x: x["text"] != "", num_proc=4)

    ds.to_parquet(opt.output_parquet)
