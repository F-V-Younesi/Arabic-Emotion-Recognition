!pip install pytorch_pretrained_bert transformers

# importing libraries:
import json, sys, regex, os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
from transformers import *
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("your device ", device)

def data_prepare_BERT(file_path, lab2ind, tokenizer, content_col, label_col, MAX_LEN):
	
	df = pd.read_excel(file_path, header=0)
	df = df[df[content_col].notnull()]
	df = df[df[label_col].notnull()]
	print("Data size ", df.shape)

	# Create sentence and label lists
	sentences = df[content_col].values
	sentences = ["[CLS] " + sentence+ " [SEP]" for sentence in sentences]
	labels = df[label_col].values
	labels = [lab2ind[i] for i in labels]

	# Import the BERT tokenizer, used to convert our text into tokens that correspond to BERT's vocabulary.
	tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
	input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
	pad_ind = tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
	input_ids = pad_sequences(input_ids, maxlen=MAX_LEN+2, dtype="long", truncating="post", padding="post", value=pad_ind)
	
        # Create attention masks
	attention_masks = []

	# Create a mask of 1s for each token followed by 0s for padding
	for seq in input_ids:
		seq_mask = [float(i > 0) for i in seq]
		attention_masks.append(seq_mask)

	# Convert all of our data into torch tensors, the required datatype for our model
	inputs = torch.tensor(input_ids)
	labels = torch.tensor(labels)
	masks = torch.tensor(attention_masks)
	return inputs, labels, masks

config={"data_file": "test20.xlsx", #data file path
             "pretrained_model_path": 'MARBERT_pytorch_verison', #MARBERT checkpoint path
             "model_path": '2-Emotion Recognition/model/model_2', #MARBERT finetuned model path
             "content_col": "tweet", #text column
             "label_col": "label", #label column
              "max_seq_length": 128, #max sequance length
              "batch_size": 16, #batch size
        }

def inference(model, iterator):
	model.eval()
	epoch_loss = 0
	all_pred=[]
	all_label = []
	with torch.no_grad():
		for i, batch in enumerate(iterator):
			# Add batch to GPU
			# batch = tuple(t for t in batch)
			batch = tuple(t.to(device) for t in batch)
			# Unpack the inputs from our dataloader
			input_ids, input_mask, labels = batch
			# outputs = model(input_ids, input_mask, labels=labels)
			outputs = model.to(device)(input_ids, input_mask, labels=labels)
			loss, logits = outputs[:2]
			# delete used variables to free GPU memory
			del batch, input_ids, input_mask
			if torch.cuda.device_count() == 1:
				epoch_loss += loss.cpu().item()
			else:
				epoch_loss += loss.sum().cpu().item()
			# identify the predicted class for each example in the batch
			probabilities, predicted = torch.max(logits.cpu().data, 1)
			# put all the true labels and predictions to two lists
			all_pred.extend(predicted)
			all_label.extend(labels.cpu())
	result=[]
	for i in range(0,len(all_pred)):
		result.append(int(all_pred[i]))
	label_list=[]
	for i in range(0,len(all_label)):
		label_list.append(int(all_label[i]))
	return  result,label_list

pre_model_path = config['pretrained_model_path']
model_path=config['model_path']
tokenizer = BertTokenizer.from_pretrained(pre_model_path)
content_col = config["content_col"]
label_col = config["label_col"]
data_file = config["data_file"]
max_seq_length= int(config["max_seq_length"])
batch_size = int(config["batch_size"])
global label2idx_file
# label2idx_file = config["data_dir"]+config["task_name"]+"_labels-dict.json"
lab2ind = json.load(open('2-Emotion Recognition/model/IraqiT123_MARBERT_labels-dict.json'))


model = BertForSequenceClassification.from_pretrained(model_path, num_labels=5)
validation_inputs, validation_labels, validation_masks = data_prepare_BERT(data_file, lab2ind, tokenizer, content_col, label_col,max_seq_length)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size)
criterion = nn.CrossEntropyLoss()
result,label_list=inference(model, validation_dataloader)
result