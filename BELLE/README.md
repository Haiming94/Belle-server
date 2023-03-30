# 教程

## 环境安装

```bash
python3 -m virtualenv -p python3 envs
source envs/bin/activate
pip install torch-1.12.1+cu113-cp38-cp38-linux_x86_64.whl 
pip install git+https://github.com/huggingface/transformers
pip install accelerate sentencepiece
pip install flask
```

## 模型下载

```bash
# 假设需要将模型下载到 `user/path/models/BELLE-LLAMA-7B-2M`
cd user/path/models/BELLE-LLAMA-7B-2M

# 将模型文件下载命令写入down.sh中
vim down.sh
# 将下面内容粘贴进去并保存
wget -c https://huggingface.co/BelleGroup/BELLE-LLAMA-7B-2M/resolve/main/pytorch_model.bin
wget https://huggingface.co/BelleGroup/BELLE-LLAMA-7B-2M/resolve/main/generation_config.json
wget https://huggingface.co/BelleGroup/BELLE-LLAMA-7B-2M/resolve/main/config.json
wget https://huggingface.co/BelleGroup/BELLE-LLAMA-7B-2M/resolve/main/special_tokens_map.json
wget https://huggingface.co/BelleGroup/BELLE-LLAMA-7B-2M/resolve/main/tokenizer.model
wget https://huggingface.co/BelleGroup/BELLE-LLAMA-7B-2M/resolve/main/tokenizer_config.json

# 运行下载
bash down.sh
```

## 运行

```python
from transformers import LlamaForCausalLM, AutoTokenizer
import torch

ckpt = 'user/path/models/BELLE-LLAMA-7B-2M'
device = torch.device('cuda')
model = LlamaForCausalLM.from_pretrained(ckpt, device_map='auto', low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(ckpt)
prompt = "Human: 写一首中文歌曲，赞美大自然 \n\nAssistant: "
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
generate_ids = model.generate(input_ids, max_new_tokens=500, do_sample = True, top_k = 30, top_p = 0.85, temperature = 0.5, repetition_penalty=1., eos_token_id=2, bos_token_id=1, pad_token_id=0)
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
response = output[len(prompt):]
```

## 服务+调用

### 服务

```python
from transformers import LlamaForCausalLM, AutoTokenizer
import torch

from flask import Flask
from flask import request
from flask import jsonify
import json


class Server7B2M:
    def __init__(self, ckpt_path):
        
        self.device = torch.device('cuda')
        self.model = LlamaForCausalLM.from_pretrained(ckpt_path, device_map='auto', low_cpu_mem_usage=True)
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    
    def serve_request(self, raw_request):
        input_ids = self.tokenizer(raw_request['prompt'], return_tensors="pt").input_ids.to(self.device)
        generate_ids = self.model.generate(
            input_ids, 
            do_sample = True, 
            max_new_tokens=raw_request['max_new_tokens'], 
            top_k = raw_request['top_k'],
            top_p = raw_request['top_p'], 
            temperature = raw_request['temperature'],
            repetition_penalty=raw_request['repetition_penalty'],
            eos_token_id=raw_request['eos_token_id'],
            bos_token_id=raw_request['bos_token_id'],
            pad_token_id=raw_request['pad_token_id'],
        )
        output = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response = output[len(prompt):]
        print(response)
        return response


ckpt = 'user/path/models/BELLE-LLAMA-7B-2M'
model_port = 5050

prompt = "Human: 写一首中文歌曲，赞美大自然 \n\nAssistant: "
raw_request = {
    'prompt': prompt,
    'max_new_tokens': 512,
    'top_k': 30,
    'top_p': 0.85,
    'temperature': 0.5,
    'repetition_penalty': 1.0,
    'eos_token_id': 2,
    'bos_token_id': 1,
    'pad_token_id': 0,
}
        
gpt_server = Server7B2M(ckpt)
print(gpt_server.serve_request(raw_request))
app = Flask(__name__)


@app.route('/func',methods=['POST','GET'])
def output_data():
    text = request.json
    dict_input = json.loads(text)
    if text:
        temp = gpt_server.serve_request(dict_input)
        return json.dumps(temp)
    else:
        return "Error input."
    
if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(host='0.0.0.0',port=model_port)  # 127.0.0.1 #指的是本地ip
```


### api调用

```python
# -*- coding: utf-8 -*-
import json
import requests



prompt = "Human: 写一首中文歌曲，赞美大自然 \n\nAssistant: "
raw_request = {
    'prompt': prompt,
    'max_new_tokens': 512,
    'top_k': 30,
    'top_p': 0.85,
    'temperature': 0.5,
    'repetition_penalty': 1.0,
    'eos_token_id': 2,
    'bos_token_id': 1,
    'pad_token_id': 0,
}

url = 'http://127.0.0.1:5050//func'
data_json = json.dumps(raw_request)
response = requests.post(url, json=data_json)
result = response.json()
print(result)
```
