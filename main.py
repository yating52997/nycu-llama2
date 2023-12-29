import os 

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from datasets import concatenate_datasets
import torch
import loralib as lora
from peft import LoraConfig, get_peft_model, PeftModel, PeftModelForCausalLM
import argparse
import wandb

os.chdir(os.path.dirname(os.path.abspath(__file__)))
torch.cuda.empty_cache()

PATH = './checkpoint'

parser = argparse.ArgumentParser()
parser.add_argument('--resume', default='', help='resume checkpoint')
parser.add_argument('--save', default='checkpoint', help='path to the folder to save checkpoint')
parser.add_argument('--export', default='export', help='path to the folder to upload to hub')
parser.add_argument('--epoch', default=10, help='number of epochs to train')
parser.add_argument('--batch_size', default=8, help='batch size')
parser.add_argument('--lr', default=1e-4, help='learning rate')
parser.add_argument('--rank', default=16, help='LoRA rank')
parser.add_argument('--alpha', default=16, help='LoRA alpha')
parser.add_argument('--test_size', default=0.05, help='test size')
parser.add_argument('--merge', default=True, help='merge model')
args = parser.parse_args()

# def get_conversation(dataset):
#     formatted_conversations = []
#     for i in range(len(dataset['conversation']) - 1):
#         if dataset['conversation'][i]['content'] != '' and dataset['conversation'][i + 1]['content'] != '':
#             formatted_conversations.append(f"<{dataset['conversation'][i]['role'].upper()}> {dataset['conversation'][i]['content']}")
#     text = ' '.join(formatted_conversations)
#     return {'text': text}


def tokenize_function(datasets):
    return tokenizer(datasets['input'],padding=True,truncation=True,max_length=512)

def tokenized_dataset(datapath):
    dataset = load_dataset('json', data_files=datapath)
    dataset = dataset['train']
    dataset = dataset.shuffle(seed=42) 
    dataset = dataset.map(tokenize_function,remove_columns=['input'])
    return dataset


if __name__ == '__main__':
    wandb.init(project="nycu-llama",name="lora config") 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('yentinglin/Taiwan-LLM-7B-v2.1-chat')
    
    # collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    qadataset = tokenized_dataset(datapath='qa.json')
    rawdataset = tokenized_dataset(datapath='raw.json')
    dataset = concatenate_datasets([qadataset,rawdataset])
    dataset = dataset.train_test_split(test_size=args.test_size, shuffle=True)
    train_ds = dataset['train']
    test_ds = dataset['test']
    
    # prepare model
    model = AutoModelForCausalLM.from_pretrained('yentinglin/Taiwan-LLM-7B-v2.1-chat',torch_dtype=torch.float16,device_map='auto')
    model.resize_token_embeddings(len(tokenizer))

    model.gradient_checkpointing_enable()
    
    # add adapters
    config = LoraConfig(
        r=args.rank, 
        lora_alpha=args.alpha,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
    )

    model = PeftModelForCausalLM(model, config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.save, #The output directory
        overwrite_output_dir=True, #overwrite the content of the output directory

        num_train_epochs=args.epoch, # number of training epochs
        per_device_train_batch_size=args.batch_size, # batch size for training
        per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        learning_rate=args.lr,
        weight_decay=1e-4,
        warmup_ratio = 0.05,
        fp16=True,
        gradient_accumulation_steps=1,
        save_total_limit=1,
        logging_steps= 100, # number of logs per epoch
        remove_unused_columns=True,
        resume_from_checkpoint=args.resume,
        gradient_checkpointing=True,
        report_to='wandb',

        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=100,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
    )

    trainer.train()

    # saving the model
    if args.merge == True:
        merge_model = model.merge_and_unload()
        if not os.path.exists(args.export):
            os.mkdir(args.export)
        
        merge_model.save_pretrained(args.export)
        tokenizer.save_pretrained(args.export)
        print("Model saved to {} waiting to upload".format(args.export))
