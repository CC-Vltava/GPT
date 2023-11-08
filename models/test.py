#!/usr/bin/env python
# coding=utf-8
from torch.utils.data import Dataset, DataLoader, random_split
import json
import argparse
import logging
import math
import torch.nn as nn
import os
import random
import datasets
import torch
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from model import GPT
import time
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    BitsAndBytesConfig,
    AutoModel,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

model = GPT()

model.load_state_dict(torch.load("/data1/cchuan/model_weight/oasst1_1M/pytorch_model.bin"))

print(model)
