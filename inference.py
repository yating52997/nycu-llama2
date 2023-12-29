from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import argparse
# PATH = os.environ['HF_CHECKPOINT']
PATH = './checkpoints'
parser = argparse.ArgumentParser()
parser.add_argument('--export', default='export', help='path to latest checkpoint')
args = parser.parse_args()


def infer(inp):
    inp = "USER: " + inp + " ASSISTANT:" 
    token = tokenizer(inp, return_tensors="pt")
    X = token["input_ids"].to(device)
    a = token["attention_mask"].to(device)
    output = model.generate(input_ids=X,attention_mask = a,pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens=256,temperature=0.2,top_p=0.95, do_sample=True, num_return_sequences=1)
    output = tokenizer.decode(output[0])
    return output.replace(inp,'')



if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.export)
    model = AutoModelForCausalLM.from_pretrained(args.export)
    model = model.to(device)
    model.eval()
    print("Welcome to ChatBot")
    while True:
        inp = input('User:')
        print(infer(inp))