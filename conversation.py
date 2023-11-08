import json
from models.model import GPT
from transformers import AutoTokenizer, AutoModel

def solve_data(samples):
    encode_tokenizer_path = '/data1/cchuan/data/weight/xlmr/'

    encode_tokenizer = AutoTokenizer.from_pretrained(encode_tokenizer_path)
    encode_tokenizer.padding_side = 'right'
    encode_max_length = 256


    print('finish data loading')

    encoded_data = encode_tokenizer(
        samples['prompt'],
        padding="max_length",
        return_tensors="pt",
        truncation=True,
        max_length=encode_max_length  # 显式指定最大长度
    )

    print('finish tokenize')

    data = {'text': encoded_data, 'answer': samples['completion']}

    return data



input_text = "Your input"


GPU_DEVICE='cuda'
# sample = read_data()
sample = {
    'prompt': [input_text],
    'completion': ['']
}

sample = solve_data(sample)

model = GPT()
model.load_model('/data1/cchuan/model_weight/GPT_1M/5/pytorch_model.bin')
model.to_device(GPU_DEVICE)

output = model.generate(sample)

print('Here is the output')
print(output)

