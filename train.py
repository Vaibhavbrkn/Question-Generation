from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
import transformers
from tqdm import tqdm
import torch
import torch.nn as nn
import argparse
import pandas as pd
import numpy as np

from sklearn import model_selection
from sklearn import metrics


my_parser = argparse.ArgumentParser()

my_parser.add_argument('device',
                       type=str,
                       help='type of device cuda/cpu')

my_parser.add_argument('max_len',
                       type=int,
                       help='Maximum Length of tokens')

my_parser.add_argument('train_batch',
                       type=int,
                       help='Training batch size')

my_parser.add_argument('val_batch',
                       type=int,
                       help='Validaation batch size')

my_parser.add_argument('epochs',
                       type=int,
                       help='number of epochs to train model')

my_parser.add_argument('data',
                       type=str,
                       help='Path of input csv file')

my_parser.add_argument('save_path',
                       type=str,
                       help='Path to save trained model')

args = my_parser.parse_args()

DEVICE = args.device
MAX_LEN = args.max_len
TRAIN_BATCH_SIZE = args.train_batch
VALID_BATCH_SIZE = args.val_batch
EPOCHS = args.epochs
MODEL_PATH = args.save_path
TRAINING_FILE = args.data


tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')
model.to(DEVICE)
dfx = pd.read_csv(TRAINING_FILE).fillna("none")


class T5Dataset:
    def __init__(self, context, question, text):
        self.tokenizer = tokenizer
        self.max_len = MAX_LEN
        self.context = context
        self.question = question
        self.text = text

    def __len__(self):
        return len(self.question)

    def __getitem__(self, item):
        context = "context: " + \
            self.context[item] + " keyword: " + self.text[item]
        question = self.question[item]

        source_tokenizer = self.tokenizer.encode_plus(
            context, max_length=MAX_LEN, pad_to_max_length=True, return_tensors="pt")

        tokenized_targets = self.tokenizer.encode_plus(
            question, max_length=MAX_LEN, pad_to_max_length=True, return_tensors="pt"
        )

        source_ids = source_tokenizer["input_ids"].squeeze()
        target_ids = tokenized_targets["input_ids"].squeeze()
        src_mask = source_tokenizer["attention_mask"].squeeze()
        target_mask = tokenized_targets["attention_mask"].squeeze()

        return {
            "source_ids": torch.tensor(source_ids, dtype=torch.long),
            "src_mask": torch.tensor(src_mask, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "target_mask": torch.tensor(target_mask, dtype=torch.long),
        }


df_train, df_valid = model_selection.train_test_split(
    dfx, test_size=0.1, random_state=42
)

df_train = df_train.reset_index(drop=True)
df_valid = df_valid.reset_index(drop=True)


train_dataset = T5Dataset(
    context=df_train.context.values, question=df_train.question.values, text=df_train.text.values
)
train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=TRAIN_BATCH_SIZE
)

valid_dataset = T5Dataset(
    context=df_valid.context.values, question=df_valid.question.values, text=df_valid.text.values
)
valid_data_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=VALID_BATCH_SIZE
)


param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_parameters = [
    {
        "params": [
            p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.001,
    },
    {
        "params": [
            p for n, p in param_optimizer if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]

num_train_steps = int(
    len(df_train) / TRAIN_BATCH_SIZE * EPOCHS)
optimizer = AdamW(optimizer_parameters, lr=3e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
)


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()

    num = 1
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        lm_labels = d["target_ids"]
        input_ids = d["source_ids"]
        attention_mask = d["src_mask"]
        decoder_attention_mask = d['target_mask']

        lm_labels = lm_labels.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        decoder_attention_mask = decoder_attention_mask.to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=None,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels
        )
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.save_pretrained(MODEL_PATH)


for i in range(EPOCHS):
    train_fn(train_data_loader, model, optimizer, DEVICE, scheduler)
