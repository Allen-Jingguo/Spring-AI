from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import load_dataset

# 模型与 tokenizer
model_path = "./Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

# 数据
# dataset = load_dataset("json", data_files="data.json")
dataset = load_dataset("csv", data_files="data1.csv")
def tokenize(example):
    tokens = tokenizer(example["text"], truncation=True, padding=True, max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
print(tokenized_dataset["train"][0])

# 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    remove_unused_columns=False,
    logging_steps=10
)

# Collator & Trainer
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
    tokenizer=tokenizer # 显式声明
)

trainer.train()
trainer.save_model("./saved_model")                  # 保存模型和 config
tokenizer.save_pretrained("./saved_model")          # 保存 tokenizer 相关文件
