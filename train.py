#!/usr/bin/env python
# coding=utf-8


import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import time

import argparse
import logging
import math
import os
import random
import datasets
import torch
from functools import partial
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from models.model import GPT
from transformers import AutoTokenizer, AutoModel


import json


def read_data():
    data_list = []

    data_path = '/data1/cchuan/data/mllm/clean_data1.json'

    with open(data_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            data_list.append(data)

    # 现在，data_list 包含了所有的 JSON 对象

    samples = {'prompt':[], 'completion': []}

    for item in data_list[0]['train']:
        samples['prompt'].append(item['input'])
        samples['completion'].append(item['output'])
    
    print('finish reading')

    return samples


def solve_data(samples, max_length):
    encode_tokenizer_path = '/data1/cchuan/data/weight/xlmr/'
    decode_tokenizer_path = '/data1/cchuan/data/weight/tiny_llama'

    encode_tokenizer = AutoTokenizer.from_pretrained(encode_tokenizer_path)
    encode_tokenizer.padding_side = 'right'
    encode_max_length = max_length

    decode_tokenizer = AutoTokenizer.from_pretrained(decode_tokenizer_path)
    decode_tokenizer.pad_token = "$$"
    decode_tokenizer.padding_side = 'right'
    decode_max_length = max_length

    print('finish loading')

    start_time = time.time()

    encoded_data = encode_tokenizer(
        samples['prompt'],
        padding="max_length",
        return_tensors="pt",
        truncation=True,
        max_length=encode_max_length,  # 显式指定最大长度
    )

    time1 = time.time()

    print('finish encode tokenize, cost time {} s'.format(time1 - start_time))

    text = [t + '\n' for t in samples["completion"]]

    decode_data = decode_tokenizer(
        text,
        padding="max_length",
        return_tensors="pt",
        truncation=True,
        max_length=decode_max_length,  # 显式指定最大长度
        add_special_tokens=False,
    )

    time2 = time.time()

    print('finish decode tokenize, cost time {} s'.format(time2 - start_time))

    print('finish tokenize')

    return {'input_ids': encoded_data['input_ids'], 'attention_mask': encoded_data['attention_mask']}, \
        {'input_ids': decode_data['input_ids'], 'attention_mask': decode_data['attention_mask']}


class CustomDataset(Dataset):
    def __init__(self, encoded_data, decode_data):
        self.encoded_data = encoded_data
        self.decode_data = decode_data

    def __len__(self):
        return len(self.encoded_data['input_ids'])

    def __getitem__(self, idx):
        encoded_text = {'input_ids': self.encoded_data['input_ids'][idx], 'attention_mask': self.encoded_data['attention_mask'][idx]}
        decode_text = {'input_ids': self.decode_data['input_ids'][idx], 'attention_mask': self.decode_data['attention_mask'][idx]}

        # 这里你可以根据需要进行数据转换、预处理等操作
        # encoded_text_tensor = torch.tensor(encoded_text)
        # decode_text_tensor = torch.tensor(decode_text)

        return {'text': encoded_text, 'answer': decode_text}


def get_dataloader(batch_size=10, max_length=256, validation_split=0.2):
    samples = read_data()
    encoded_data, decode_data = solve_data(samples, max_length)
    custom_dataset = CustomDataset(encoded_data, decode_data)
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    dataset = data_loader.dataset
    dataset_size = len(dataset)
    validation_size = int(dataset_size * validation_split)
    split_sizes = [dataset_size - validation_size, validation_size]
    train_dataset, val_dataset = random_split(dataset, split_sizes)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader


def count(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


train_dataloader, val_dataloader = get_dataloader()
model = GPT()
model.to_device('cuda')
print('Finish Model')


print('Number of all parameters')
print(str(count(model)))
print('Number of Q-Former')
print(count(model.Qformer))
print('Number of LLAMA Projection')
print(count(model.llama_proj))


optimizer = optim.AdamW(model.parameters(), lr=5e-5)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for data in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
    # for data in data_loader:
        # 清除之前的梯度

        optimizer.zero_grad()

        # 前向传播
        loss = model(data)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for data in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                loss = model(data)
                total_loss += loss
            print("Total loss is {}".format(total_loss))


# 保存训练好的模型
torch.save(model.state_dict(), 'my_model.pth')
