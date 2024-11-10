# Arabic-Emotion-Recognition
Emotion Recognition Repo for Iraqi Tweets.


# Project tree
 * [emo_rec](./API)
   * [static](./API)
   * [templates](./API)
     * [index.html](./API)
   * [Marbert_Inference_API.py](./API)
   * [requirements](./API)
   * [app.py](./API)

As shown above, you must create a directory named "emo_rec" with files and folders with same above names in your working directory.
take attention: don't change directory(cd) to "emo_rec".

## For Inference:

1-install necessary packages:
```
!pip install -r requirements.txt
# or in conda: conda install --yes --file requirements.txt
```
2- download pretrained model from: [model](https://huggingface.co/UBC-NLP/MARBERT/blob/main/MARBERT_pytorch_verison.tar.gz) </br>
```
!wget https://huggingface.co/UBC-NLP/MARBERT/resolve/main/MARBERT_pytorch_verison.tar.gz
!tar -xvf MARBERT_pytorch_verison.tar.gz
<br>
```
3- download FineTune model (trained on dataset3-1) and config : [model](https://huggingface.co/fvyounesi/Marbert_Iraqi_FineTuned) </br>
4- download label-dict file from here: [label-dict](https://huggingface.co/fvyounesi/Marbert_Iraqi_FineTuned) </br>
5- run Marbert_inference.py
