import pandas as pd

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Hugging Face repo name, NOT local path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

def sentiment(text):
    # prompt = (
    #     "Based on the text below:\n"
    #     "1. What is the expected trend? Use -1 for negative, 0 for neutral, 1 for positive.\n"
    #     "Respond with only a number, No words. Example: 1\n\n"
    #     + text
    # )
    prompt = 'What is the expected trend according to this text ? Answer negative or positive \n\n' + text
    # prompt = 'What is the expected trend according to this text ? \n\n' + text

    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    outputs = model.generate(**inputs, max_new_tokens=2048)
    # outputs = model.generate(**inputs)

    human_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return human_output


current_dir = Path(__file__).parent
reports = pd.read_parquet(current_dir / 'mda_text.parquet')
print(reports)
print(reports.head())

# for report in reports:

text = str(reports.iloc[0]['text'])[:2048]
# text = str(reports.iloc[0])
# text = 'This is going to the moon! very confident it will skyrocket'
# text = 'This is going to crash! Not likly it will survive'

human_output = sentiment(text)
print(human_output)
