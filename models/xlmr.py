from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn as nn

class xlmr():
	def __init__(self):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.tokenizer = AutoTokenizer.from_pretrained("/data1/cchuan/data/weight/xlmr/")
		self.model = AutoModelForMaskedLM.from_pretrained\
			("/data1/cchuan/data/weight/xlmr/").to(self.device)
		self.num_features = 250002

	def freeze(self):
		for param in self.model.parameters():
			param.requires_grad = False
		self.model.eval()

	def unfreeze(self):
		for param in self.model.parameters():
			param.requires_grad = True

	def to(self, device):
		self.device = device
		self.model.to(device)

	def __call__(self, encoded_data):
		input_ids = encoded_data["input_ids"].to(self.device)
		attention_mask = encoded_data["attention_mask"].to(self.device)
		features = self.model(input_ids, attention_mask)
		return features.logits.to(self.device)


# xlm = xlmr()
# # xlm.to_gpu()
# # print(xlm.extract_features('hello world!').size())

# tokenizer = AutoTokenizer.from_pretrained("/data1/cchuan/data/weight/xlmr/")
# text = tokenizer('hello world', return_tensors='pt').to('cuda')
# print(text)
# print(xlm(text).size())
