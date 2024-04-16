print("testingggg")

import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
print("testingggg2")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", load_in_4bit=True, torch_dtype=torch.float16, device_map="auto")
print("testingggg3")

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)
tokenizer.pad_token = "!"
CUTOFF_LEN = 256
LORA_R = 8
LORA_ALPHA = 2 * LORA_R
LORA_DROPOUT = 0.1


config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=[ "w1", "w2", "w3"], lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, config)
dataset = load_dataset("harpreetsahota/modern-to-shakesperean-translation")
print("dataset", dataset)
train_data = dataset["train"]


def generate_prompt(user_query):
  sys_msg = "Translate the given text to Shakespearean style."
  p = "<s> [INST]" + sys_msg +"\n"+ user_query["modern"] + "[/INST]" +  user_query["shakespearean"] + "</s>"
  return p 


tokenize = lambda prompt: tokenizer(prompt + tokenizer.eos_token, truncation=True, max_length=CUTOFF_LEN, padding="max_length")
train_data = train_data.shuffle().map(lambda x: tokenize(generate_prompt(x)), remove_columns=["modern" , "shakespearean"])


trainer = Trainer(
  model=model,
  train_dataset=train_data,
  args=TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=6,
    learning_rate=1e-4,
    logging_steps=2,
    optim="adamw_torch",
    save_strategy="epoch",
    output_dir="mixtral-moe-lora-instruct-shapeskeare"
  ),
  data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)


model.config.use_cache = False
trainer.train()