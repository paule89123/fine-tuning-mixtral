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

CUTOFF_LEN = 128  # Reduced from 256
LORA_R = 1  # Reduced from 8
LORA_ALPHA = 2 * LORA_R
LORA_DROPOUT = 0.1

config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=[
                    "w1", "w2", "w3"], lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, config)

dataset = load_dataset("Biddls/Onion_News")
print("dataset", dataset)

train_data = dataset["train"]

# Calculate the number of samples to use (5% of the dataset)
num_samples = int(len(train_data) * 0.02)

# Shuffle the dataset and select the first 5% of samples
train_data = train_data.shuffle().select(range(num_samples))

def generate_prompt(sample):
    sys_msg = "Write an article in the style of The Onion based on the given headline and article text."
    headline, article = sample["text"].split("#~#")
    p = "<s> [INST]" + sys_msg + "\n" + headline.strip() + "[/INST]" + article.strip() + "</s>"
    return p

tokenize = lambda prompt: tokenizer(prompt + tokenizer.eos_token, truncation=True, max_length=CUTOFF_LEN, padding="max_length")
train_data = train_data.map(lambda x: tokenize(generate_prompt(x)), remove_columns=["text"])

trainer = Trainer(
    model=model,
    train_dataset=train_data,
    args=TrainingArguments(
        per_device_train_batch_size=4,  # Increased from 1
        gradient_accumulation_steps=2,  # Reduced from 4
        num_train_epochs=1,  # Reduced from 6
        learning_rate=1e-4,
        logging_steps=2,
        optim="adamw_torch",
        save_strategy="epoch",
        output_dir="mixtral-moe-lora-instruct-onion"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

model.config.use_cache = False
trainer.train()