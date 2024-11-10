# importing libraries:
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
		text = request.form['textarea']
		listtext=[text]
		df=pd.DataFrame(listtext, columns=["tweet"])
		file_path = "emo_rec/static/input_data.xlsx"
		df.to_excel(file_path)
		p=prediction()

	return render_template("index.html", prediction = p, file_path = file_path)


if __name__ =='__main__':
	# app.debug = True
	app.run(debug = True)
