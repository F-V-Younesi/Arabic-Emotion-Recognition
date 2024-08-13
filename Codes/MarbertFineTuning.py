# importing libraries:
import json, sys, regex
import torch
import GPUtil
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, confusion_matrix
##----------------------------------------------------
from transformers import *
from transformers import XLMRobertaConfig
from transformers import XLMRobertaModel
from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, XLMRobertaModel
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel



def create_label2ind_file(file, label_col):
	labels_json={}
	#load train_dev_test file
	# df = pd.read_csv(file, sep="\t")
	df = pd.read_excel(file)
	# df = pd.read_csv(file, sep="\t",encoding = 'utf-8', encoding_errors='ignore')
	df.head(5)
	#get labels and sort it A-Z
	labels = df[label_col].unique()
	labels.sort()
	#convert labels to indexes
	for idx in range(0, len(labels)):
		labels_json[labels[idx]]=idx
	#save labels with indexes to file
	with open(label2idx_file, 'w') as json_file:
		json.dump(labels_json, json_file)


def data_prepare_BERT(file_path, lab2ind, tokenizer, content_col, label_col, MAX_LEN):
	# Use pandas to load dataset
	# df = pd.read_csv(file_path, delimiter='\t', header=0, encoding = 'utf-8', encoding_errors='ignore')
	# df = pd.read_csv(file_path, delimiter='\t', header=0)
	df = pd.read_excel(file_path, header=0)
	df = df[df[content_col].notnull()]
	df = df[df[label_col].notnull()]
	print("Data size ", df.shape)
	# Create sentence and label lists
	sentences = df[content_col].values
	sentences = ["[CLS] " + sentence+ " [SEP]" for sentence in sentences]
	print ("The first sentence:")
	print (sentences[0])
	# Create sentence and label lists
	labels = df[label_col].values
	#print (labels)
	labels = [lab2ind[i] for i in labels]
	# Import the BERT tokenizer, used to convert our text into tokens that correspond to BERT's vocabulary.
	tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
	print ("Tokenize the first sentence:")
	print (tokenized_texts[0])
	#print("Label is ", labels[0])
	# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
	input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
	print ("Index numbers of the first sentence:")
	print (input_ids[0])
	# Pad our input seqeunce to the fixed length (i.e., max_len) with index of [PAD] token
	# ~ input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
	pad_ind = tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
	input_ids = pad_sequences(input_ids, maxlen=MAX_LEN+2, dtype="long", truncating="post", padding="post", value=pad_ind)
	print ("Index numbers of the first sentence after padding:\n",input_ids[0])
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

def flat_pred(preds, labels):
	pred_flat = np.argmax(preds, axis=1).flatten()
	labels_flat = labels.flatten()
	return pred_flat.tolist(), labels_flat.tolist()


def train(model, iterator, optimizer, scheduler, criterion):

	model.train()
	epoch_loss = 0
	for i, batch in enumerate(iterator):
		# Add batch to GPU
		batch = tuple(t.to(device) for t in batch)
		# Unpack the inputs from our dataloader
		input_ids, input_mask, labels = batch
		outputs = model(input_ids, input_mask, labels=labels)
		loss, logits = outputs[:2]
		# delete used variables to free GPU memory
		del batch, input_ids, input_mask, labels
		optimizer.zero_grad()
		if torch.cuda.device_count() == 1:
			loss.backward()
			epoch_loss += loss.cpu().item()
		else:
			loss.sum().backward()
			epoch_loss += loss.sum().cpu().item()
		optimizer.step()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore
		# optimizer.step()
		scheduler.step()
	# free GPU memory
	if device == 'cuda':
		torch.cuda.empty_cache()
	return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
	model.eval()
	epoch_loss = 0
	all_pred=[]
	all_label = []
	with torch.no_grad():
		for i, batch in enumerate(iterator):
			# Add batch to GPU
			batch = tuple(t.to(device) for t in batch)
			# Unpack the inputs from our dataloader
			input_ids, input_mask, labels = batch
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
	# return  all_pred,outputs
	accuracy = accuracy_score(all_label, all_pred)
	f1score_ma = f1_score(all_label, all_pred, average='macro')
	f1score_w = f1_score(all_label, all_pred, average='weighted')
	recall = recall_score(all_label, all_pred, average='macro')
	precision = precision_score(all_label, all_pred, average='macro')
	reportc = classification_report(all_label, all_pred)
	print(reportc)
	return (epoch_loss / len(iterator)), accuracy, f1score_ma,f1score_w, recall, precision

def fine_tuning(config):
	#---------------------------------------
	print ("[INFO] step (1) load train_test config file")
	# config_file = open(config_file, 'r', encoding="utf8")
	# config = json.load(config_file)
	task_name = config["task_name"]
	content_col = config["content_col"]
	label_col = config["label_col"]
	train_file = config["data_dir"]+config["train_file"]
	dev_file = config["data_dir"]+config["dev_file"]
	sortby = config["sortby"]
	max_seq_length= int(config["max_seq_length"])
	batch_size = int(config["batch_size"])
	lr_var = float(config["lr"])
	model_path = config['pretrained_model_path']
	num_epochs = config['epochs'] # Number of training epochs (authors recommend between 2 and 4)
	global label2idx_file
	label2idx_file = config["data_dir"]+config["task_name"]+"_labels-dict.json"
	#-------------------------------------------------------
	print ("[INFO] step (2) convert labels2index")
	create_label2ind_file(train_file, label_col)
	print (label2idx_file)
	#---------------------------------------------------------
	print ("[INFO] step (3) check checkpoit directory and report file")
	ckpt_dir = config["data_dir"]+task_name+"_bert_ckpt/"
	report = ckpt_dir+task_name+"_report.tsv"
	sorted_report = ckpt_dir+task_name+"_report_sorted.tsv"
	if not os.path.exists(ckpt_dir):
		os.mkdir(ckpt_dir)
	#-------------------------------------------------------
	print ("[INFO] step (4) load label to number dictionary")
	lab2ind = json.load(open(label2idx_file))
	print ("[INFO] train_file", train_file)
	print ("[INFO] dev_file", dev_file)
	print ("[INFO] num_epochs", num_epochs)
	print ("[INFO] model_path", model_path)
	print ("max_seq_length", max_seq_length, "batch_size", batch_size)
	#-------------------------------------------------------
	print ("[INFO] step (5) Use defined funtion to extract tokanize data")
	# tokenizer from pre-trained BERT model
	print ("loading BERT setting")
	tokenizer = BertTokenizer.from_pretrained(model_path)
	train_inputs, train_labels, train_masks = data_prepare_BERT(train_file, lab2ind, tokenizer,content_col, label_col, max_seq_length)
	validation_inputs, validation_labels, validation_masks = data_prepare_BERT(dev_file, lab2ind, tokenizer, content_col, label_col,max_seq_length)
	# Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top.
	model = BertForSequenceClassification.from_pretrained(model_path, num_labels=len(lab2ind))
	#--------------------------------------
	print ("[INFO] step (6) Create an iterator of data with torch DataLoader.")
#		  This helps save on memory during training because, unlike a for loop,\
#		  with an iterator the entire dataset does not need to be loaded into memory")
	train_data = TensorDataset(train_inputs, train_masks, train_labels)
	train_dataloader = DataLoader(train_data, batch_size=batch_size)
	#---------------------------
	validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
	validation_dataloader = DataLoader(validation_data, batch_size=batch_size)
	#------------------------------------------
	print ("[INFO] step (7) run with parallel GPUs")
	if torch.cuda.is_available():
		if torch.cuda.device_count() == 1:
			print("Run", "with one GPU")
			model = model.to(device)
		else:
			n_gpu = torch.cuda.device_count()
			print("Run", "with", n_gpu, "GPUs with max 4 GPUs")
			device_ids = GPUtil.getAvailable(limit = 4)
			torch.backends.cudnn.benchmark = True
			model = model.to(device)
			model = nn.DataParallel(model, device_ids=device_ids)
	else:
		print("Run", "with CPU")
		model = model
	#---------------------------------------------------
	print ("[INFO] step (8) set Parameters, schedules, and loss function")
	global max_grad_norm
	max_grad_norm = 1.0
	warmup_proportion = 0.1
	num_training_steps	= len(train_dataloader) * num_epochs
	num_warmup_steps = num_training_steps * warmup_proportion
	### In Transformers, optimizer and schedules are instantiated like this:
	# Note: AdamW is a class from the huggingface library
	# the 'W' stands for 'Weight Decay"
	optimizer = AdamW(model.parameters(), lr=lr_var, correct_bias=False)
	# schedules
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler
	# We use nn.CrossEntropyLoss() as our loss function.
	criterion = nn.CrossEntropyLoss()
	#---------------------------------------------------
	print ("[INFO] step (9) start fine_tuning")
	for epoch in trange(num_epochs, desc="Epoch"):
		train_loss = train(model, train_dataloader, optimizer, scheduler, criterion)
		val_loss, val_acc, val_f1_ma,val_f1_w ,val_recall, val_precision= evaluate(model, validation_dataloader, criterion)
# 		print (train_loss, val_acc)
		# Create checkpoint at end of each epoch
		if not os.path.exists(ckpt_dir + 'model_' + str(int(epoch + 1)) + '/'): os.mkdir(ckpt_dir + 'model_' + str(int(epoch + 1)) + '/')
		model.save_pretrained(ckpt_dir+ 'model_' + str(int(epoch + 1)) + '/')
		epoch_eval_results = {"epoch_num":int(epoch + 1),"train_loss":train_loss,
					  "val_acc":val_acc, "val_recall":val_recall, "val_precision":val_precision, "val_f1_ma":val_f1_ma,"val_f1_w":val_f1_w,"lr":lr_var }
		with open(report,"a") as fOut:
			fOut.write(json.dumps(epoch_eval_results)+"\n")
			fOut.flush()
		#------------------------------------
		report_df = pd.read_json(report, orient='records', lines=True)
		report_df.sort_values(by=[sortby],ascending=False, inplace=True)
		report_df.to_csv(sorted_report,sep="\t",index=False)
	return report_df

df=pd.read_excel('Emotion Recognition/Dataset/dataset12.xlsx')
!mkdir -p './IraqiT'
train1, test=train_test_split(df,test_size=0.1, random_state=42,shuffle=True)
train1.to_excel('/content/IraqiT/traindata12.xlsx')
test.to_excel('/content/IraqiT/testdata12.xlsx')

config={"task_name": "IraqiT12_MARBERT", #output directory name
             "data_dir": "./IraqiT/", #data directory
             "train_file": "traindata12.xlsx", #train file path
             "dev_file": "testdata12.xlsx", #dev file path or test file path
             "pretrained_model_path": 'MARBERT_pytorch_verison', #MARBERT checkpoint path
             "epochs": 5, #number of epochs
             "content_col": "tweet", #text column
             "label_col": "label", #label column
             "lr": 9e-06, #2e-06, #learning rate
              "max_seq_length": 128, #max sequance length
              "batch_size": 16, #batch shize
              "sortby":"val_f1_w"} #sort results based on val_acc or val_f1

report_df = fine_tuning(config)
report_df