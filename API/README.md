
# Project tree

 * [static](./static)
 * [templates](./templates)
   * [index.html](./templates/index.html)
 * [AudioKeys.py](./AudioKeys.py)
 * [requirements](./requirements)

As shown above, the directory with files and folders with same names must be created.

**First: install necessary packages:**
```
!pip install -r requirements.txt
# or in conda: conda install --yes --file requirements.txt
!pip install git+https://github.com/openai/whisper.git
!python -m perke download
!python3 -m pip install -U git+https://github.com/facebookresearch/demucs#egg=demucs

#only in conda environment it is necessary to run:
!conda install conda-forge::ffmpeg
```

**Second: make following directory in local path of "parsivar package" then download and copy onegram.pckl & mybigram_lm.pckl models in this directoy. this stage must be done "after installing Packages":** <br>
download onegram.pckl from:<br>
https://drive.google.com/file/d/1-BWmc5-kH637ZpgI-DPonwpwKKk1Q8Rj/view?usp=sharing<br>
download mybigram_lm.pckl from:<br>
https://drive.google.com/file/d/1uCh2S2wqUbTke5TH7cv9V3xLILlLsGW0/view?usp=sharing<br>

```
#Making "spell" folder in "parsivar/resource" path:
!mkdir '/usr/local/lib/python3.10/dist-packages/parsivar/resource/spell' #change this with your parsivar installed path
!cp 'onegram.pckl' '/usr/local/lib/python3.10/dist-packages/parsivar/resource/spell'
!cp 'mybigram_lm.pckl' '/usr/local/lib/python3.10/dist-packages/parsivar/resource/spell'
```
**Third: Importing Libraries:**
```
import moviepy.editor
import io,os, re
from pathlib import Path
import select
import subprocess as sp
import sys
from typing import Dict, Tuple, Optional, IO
from pydub import AudioSegment, silence
import librosa #, numpy as np
from parsivar import SpellCheck
import whisper
import numpy as np
import shutil
def dummy_npwarn_decorator_factory():
  def npwarn_decorator(x):
    return x
  return npwarn_decorator
np._no_nep50_warning = getattr(np, '_no_nep50_warning', dummy_npwarn_decorator_factory)
from perke.unsupervised.graph_based import TopicRank
from flask import Flask, render_template, request
# replace "AudioKeys" name in below line with api code version name, such as "AudioKeys_5M"
from AudioKeys import *
```
As said in the comment above, replace "AudioKeys" name with api code version name, such as "AudioKeys_5M"

**Versions:**
1. **6Gold_spleeter_mp3** : The best model(in accuracy) for the case where the presence of useful content and music in the input is not known.
2. **5Gold_spleeter** : The best model(in accuracy) for the case where the presence of music in the input is not known but we know that inuputs certainly have useful contents, for example: Audio of lectures or meetings.
3. **5Medium** : The best model(in accuracy) for the case where the presence of useful content in the input is known and input has not music in background. also can use for upper cases with less accuracy but faster implementation.
4. **4Light** : good for all of cases, but has lowest accuracy but fastest in implemetation.


and then run app:
```
app = Flask(__name__)

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		file = request.files['myfile']
		text = request.form['textarea']
		file_path = "static/" + file.filename	
		file.save(file_path)
		p = predict(file_path,text)
	return render_template("index.html", prediction = p, file_path = file_path)


if __name__ =='__main__':
	# app.debug = True
	app.run(debug = True)

```
