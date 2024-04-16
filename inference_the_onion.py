import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Load the fine-tuned model and tokenizer
model_path = "./mixtral-moe-lora-instruct-onion/checkpoint-85"
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, model_path)

# Set the pad token
tokenizer.pad_token = "!"

# Function to generate an article based on a given headline
def generate_article(headline):
    sys_msg = "Write an article in the style of The Onion based on the given headline."
    prompt = f"<s> [INST]{sys_msg}\n{headline}[/INST]"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256, padding="max_length")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        num_beams=4,
        temperature=0.8,
        no_repeat_ngram_size=3,
        early_stopping=True,
        do_sample=True
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    article = generated_text.split("[/INST]")[-1].strip()
    return article

# Example usage
headline = "Instructor Bes's Attempt to Explain Neural Networks Through Interpretive Dance Leaves Students More Confused"
generated_article = generate_article(headline)
print("Headline:", headline)
print("Generated Article:")
print(generated_article)