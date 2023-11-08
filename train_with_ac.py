# 安装需要的库
# !pip install transformers datasets torch accelerate

# 导入必要的库
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from accelerate import Accelerator

# 数据加载
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
train_dataset = dataset["train"].map(lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True), batched=True)
test_dataset = dataset["test"].map(lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True), batched=True)

# 模型加载
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 训练参数
training_args = TrainingArguments(
    output_dir="./imdb",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="steps",
    save_steps=500,
    eval_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    push_to_hub=False,
    remove_unused_columns=False,
    report_to="tensorboard",
)

# 初始化 Accelerator
accelerator = Accelerator()

# 在模型和数据上使用 Accelerator
model, train_dataset, test_dataset = accelerator.prepare(model, train_dataset, test_dataset)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=tokenizer.data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# 训练
trainer.train()

# 评估
trainer.evaluate()

# 保存模型
trainer.save_model()

# 上传模型到 Hugging Face Hub
if training_args.push_to_hub:
    trainer.push_to_hub()
