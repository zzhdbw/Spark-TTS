import requests
import soundfile as sf
import json
import numpy as np

url = "http://localhost:8000/v2/models/infer_pipeline/infer"
wav_path = "*********"
waveform, sr = sf.read(wav_path)
lang_id = 54
samples = np.array([waveform], dtype=np.float32)
lengths = np.array([[len(waveform)]], dtype=np.int32)
lang_id = np.array([[lang_id]], dtype=np.int8)

data = {
    "inputs":[
        {
            "name": "WAV",
            "shape": samples.shape,
            "datatype": "FP32",
            "data": samples.tolist()
        },
        {
            "name": "WAV_LENS",
            "shape": lengths.shape,
            "datatype": "INT32",
            "data": lengths.tolist(),
        },
        {
            "name": "LANG_ID",
            "shape": lang_id.shape,
            "datatype": "INT8",
            "data": lang_id.tolist()
        }
    ]
}
rsp = requests.post(
    url,
    headers={"Content-Type": "application/json"},
    json=data,
    verify=False,
    params={"request_id": '0'}
)
result = rsp.json()
print(result)
transcripts = result["outputs"][0]["data"][0]
print(transcripts)