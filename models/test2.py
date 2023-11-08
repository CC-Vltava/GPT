import torch
from transformers import AutoTokenizer, AutoModel

# 载入BERT模型和tokenizer
model_name = "/data1/cchuan/data/weight/tiny_llama/"  # 选择你想要使用的BERT模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)



# print(model.embedding)

# 输入文本
text = ["Good Morning", "my name is cc"]

# Tokenize文本
tokenizer.pad_token = "$$"
tokens = tokenizer(text, return_tensors="pt", padding=True)


print(tokens['input_ids'].shape)

# 获取模型的嵌入
with torch.no_grad():
    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state  # BERT的嵌入通常在最后一层

# 现在，'embeddings' 包含了你的输入文本的嵌入表示

print(embeddings.shape)

# reverse = model.embed_tokens(embeddings)

matrix = model.embed_tokens.weight

reverse = embeddings @ matrix.T

print(reverse)
