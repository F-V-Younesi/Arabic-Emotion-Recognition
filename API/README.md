
# Project tree

 * [static](./static)
 * [templates](./templates)
   * [index.html](./templates/index.html)
 * [Marbert_Inference_API.py](./Marbert_Inference_API.py)
 * [requirements](./requirements)

As shown above, the directory with files and folders with same names must be created.

**First: install necessary packages:**
```
!pip install -r requirements.txt
```

**Second: make following directory in local path:** <br>
```
#mkdir "finetuned_model" folder and download fine-tuned model in "localpath/finetuned_model":
!mkdir finetuned_model
!gdown 1-5NC6uouHwsdgBHGZAskfS1lFfV-h-3s -O finetuned_model
#download config file in "localpath/finetunedmodel":
!gdown 1-5DV3IHm2vl7rGaLidP8c7dYWo8I0nOe
#download labels dictionary in local path:
!gdown 1-1QLmgsT6lc5E66ErE2e5maRfQl_7FUf
#and download pretrained model in local path:
!wget https://huggingface.co/UBC-NLP/MARBERT/resolve/main/MARBERT_pytorch_verison.tar.gz
!tar -xvf MARBERT_pytorch_verison.tar.gz
```
**Third: Run app.py:**
```
import json, sys, regex
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import os
import numpy as np
from transformers import *
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
from flask import Flask, render_template, request
from Marbert_Inference_API import *

app = Flask(__name__)

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		file = request.files['myfile']
		file_path = "static/input_data.xlsx"
		file.save(file_path)
		p=prediction()

	return render_template("index.html", prediction = p, file_path = file_path)


if __name__ =='__main__':
	# app.debug = True
	app.run(debug = True)

```
