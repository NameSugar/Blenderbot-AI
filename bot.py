# you need install this 
# also this AI can make mistake and think what he human
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained('facebook/blenderbot-400M-distill')
model = AutoModelForSeq2SeqLM.from_pretrained('facebook/blenderbot-400M-distill')

def generate_response(input_text):
    # Encode the input text
    inputs = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=512,
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    # Generate a response
    output = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=50,
        num_beams=4,
        no_repeat_ngram_size=2,
        early_stopping=True,
        temperature=1.2,
        top_k=50,
        top_p=0.95
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response

while True:
    user_input = input("User: ")
    response = generate_response(user_input)
    print("AI: ", response)
    print("(Machine learning model, may make mistakes.)\n")
