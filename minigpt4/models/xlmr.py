from transformers import AutoTokenizer, AutoModelForMaskedLM, LlamaTokenizer
import torch
import torch.nn as nn

class xlmr(nn.Module):
	def __init__(self):
		super(xlmr, self).__init__()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.tokenizer = AutoTokenizer.from_pretrained("/data1/cchuan/data/weight/xlmr/")
		self.model = AutoModelForMaskedLM.from_pretrained\
			("/data1/cchuan/data/weight/xlmr/").to(self.device)
		self.num_features = 250002

	def extract_features(self, text):
		# encoded_input = self.tokenizer(text, return_tensors='pt').to(self.device)
		features = self.model(**text)
		return features.logits
       
	def freeze(self):
		for param in self.model.parameters():
			param.requires_grad = False
		self.model.eval()

	def unfreeze(self):
		for param in self.model.parameters():
			param.requires_grad = True

	def to_cpu(self):
		self.device = 'cpu'
		
	def to_gpu(self):
		self.device = 'cuda'

	def forward(self, text):
		return self.extract_features(text)
    # def forward(self, text):
        

# xlm = xlmr()
# xlm.to_gpu()
# print(xlm.extract_features('hello world!').size())
llama_model = '/home/cchuan/Project/MiniGPT-4/weight/vicuna/'
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model)
