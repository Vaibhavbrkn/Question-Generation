# Longshot-AI


## ✨  [Dataset](https://www.kaggle.com/ananthu017/squad-csv-format)
## ✨  Installation
``` python
pip install -r requirements.txt
```
## ✨  Training 
``` python
python train.py device max_len train_batch val_batch epochs data save_path
```

## ✨  Inference
``` python
python app.py
```
## ✨   [Live Demo](https://huggingface.co/spaces/Vaibhavbrkn/Question-gen)

## 🚀 Approach 
1. T5 is used as sequence generation model
2. Dataset was cleaned to only include context, Question, text (title) from above dataset
3. prompt programming for model input : "context " + context + " keyword " + keyword
4. for inference programm first generate keywords using keybert then use this keyword with context to generate question.
5. Gradio is used as UI.
