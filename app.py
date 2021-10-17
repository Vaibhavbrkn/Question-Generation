import gradio as gr
import numpy as np
from keybert import KeyBERT
import random
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    BertTokenizer, BertForSequenceClassification
)
import re
import transformers
import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 512

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained(
    'Vaibhavbrkn/question-gen')
BertModel = BertForSequenceClassification.from_pretrained('good_bad')
BertToken = BertTokenizer.from_pretrained('good_bad')
mod = KeyBERT('distilbert-base-nli-mean-tokens')
model.to(DEVICE)
BertModel.to(DEVICE)
model.eval()
BertModel.eval()


def filter_keyword(data, ran=5):
    ap = []
    real = []
    res = re.sub(r'-', ' ', data)
    res = re.sub(r'[^\w\s\.\,]', '', res)
    for i in range(1, 4):
        ap.append(mod.extract_keywords(
            res, keyphrase_ngram_range=(1, i), diversity=0.7, top_n=ran*2))
    for i in range(3):
        for j in range(len(ap[i])):
            if ap[i][j][0].lower() in res.lower():
                real.append(ap[i][j])

    real = sorted(real, key=lambda x: x[1], reverse=True)
    ap = []
    st = ""
    for i in range(len(real)):
        if real[i][0] in st:
            continue
        else:
            ap.append(real[i])
            st += real[i][0] + " "
        if len(ap) == ran:
            break

    return ap


# FOR BAD label negative or bottom 3
labels = ['BAD', 'GOOD']


def func(context, slide):
    slide = int(slide)
    randomness = 0.4
    orig = int(np.ceil(randomness * slide))
    temp = slide - orig
    ap = filter_keyword(context, ran=slide*2)
    answer = []
    for i in range(orig):

        inputs = "context: " + context + " keyword: " + ap[i][0]
        source_tokenizer = tokenizer.encode_plus(
            inputs, max_length=512, pad_to_max_length=True, return_tensors="pt")
        outs = model.generate(input_ids=source_tokenizer['input_ids'].to(
            DEVICE), attention_mask=source_tokenizer['attention_mask'].to(DEVICE), max_length=50)
        dec = [tokenizer.decode(ids) for ids in outs][0]
        st = dec.replace("<pad> ", "")
        st = st.replace("</s>", "")
        st_token = BertToken.encode_plus(
            st, max_length=128, pad_to_max_length=True, return_tensors="pt")
        outputs = BertModel(st_token['input_ids'].to(
            DEVICE), attention_mask=st_token['attention_mask'].to(DEVICE)).logits
        outputs = outputs.detach().cpu()
        preds = torch.nn.functional.softmax(outputs, dim=1)

        answer.append((st, labels[torch.argmax(preds)]))

    del ap[: orig]

    if temp > 0:
        for i in range(temp):
            keyword = random.choice(ap)
            inputs = "context: " + context + \
                " keyword: " + keyword[0]
            source_tokenizer = tokenizer.encode_plus(
                inputs, max_length=512, pad_to_max_length=True, return_tensors="pt")
            outs = model.generate(input_ids=source_tokenizer['input_ids'].to(
                DEVICE), attention_mask=source_tokenizer['attention_mask'].to(DEVICE), max_length=50)
            dec = [tokenizer.decode(ids) for ids in outs][0]
            st = dec.replace("<pad> ", "")
            st = st.replace("</s>", "")
            st_token = BertToken.encode_plus(
                st, max_length=128, pad_to_max_length=True, return_tensors="pt")
            outputs = BertModel(st_token['input_ids'].to(
                DEVICE), attention_mask=st_token['attention_mask'].to(DEVICE)).logits
            outputs = outputs.detach().cpu()
            preds = torch.nn.functional.softmax(outputs, dim=1)
            answer.append((st, labels[torch.argmax(preds)]))

    return answer


gr.Interface(func,
             [
                 gr.inputs.Textbox(lines=10, label="context"),
                 gr.inputs.Slider(minimum=1, maximum=5,
                                  default=3, label="No of Question"),
             ],
             gr.outputs.KeyValues(), capture_session=True, server_name="0.0.0.0").launch()
