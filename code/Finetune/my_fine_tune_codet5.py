import re
import os
import pprint
import argparse
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer

# 设置环境变量，指定使用 GPU 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

PROMPT_DICT = {
    "prompt_input": (
        "Summarize the function and suggest a more descriptive function name:\n\n"
        "{code}\n\n"
        "### Summary and Suggested Name:"
    ),
}

def run_training(args, model, train_data):
    print(f"Starting main loop")

    training_args = TrainingArguments(
        report_to='tensorboard',
        output_dir=args.save_dir,
        overwrite_output_dir=False,

        do_train=True,
        save_strategy='epoch',

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,

        learning_rate=args.lr,
        weight_decay=0.05,
        warmup_steps=args.lr_warmup_steps,

        logging_dir=args.save_dir,
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_total_limit=1,

        dataloader_drop_last=True,
        dataloader_num_workers=4,

        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        fp16=args.fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
    )

    trainer.train(resume_from_checkpoint=True)

    if args.local_rank in [0, -1]:
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")
        model.save_pretrained(final_checkpoint_dir)
        print(f'  ==> Finish training and save to {final_checkpoint_dir}')


def load_tokenize_data(args):
    # Load and tokenize data
    if os.path.exists(args.cache_data):
        train_data = load_from_disk(args.cache_data)
        print(f'  ==> Loaded {len(train_data)} samples')
        return train_data
    else:
        # Load CodeSearchNet Java dataset
        data_path = args.instruct_data_path
        datasets = load_dataset('json', data_files={"train": f"{data_path}/train/*.jsonl.gz"})
        # Use the training split for fine-tuning
        train_data = datasets["train"]
        tokenizer = AutoTokenizer.from_pretrained(args.load)

        def preprocess_function(examples):
            prompt = PROMPT_DICT["prompt_input"]

            # Function to replace the function name with "unknown_func"
            def replace_function_name(func_code, func_name):
                # Extract the class name and function name from func_name
                if '.' in func_name:
                    class_name, function_name = func_name.split('.')
                else:
                    class_name = None
                    function_name = func_name

                # If there is a class name prefix, remove the first occurrence of the class name and any trailing spaces
                if class_name:
                    # Match the class name followed by any spaces
                    pattern = r'\b' + re.escape(class_name) + r'\s*'
                    # Replace with a placeholder to ensure only the first match is replaced
                    func_code = re.sub(pattern, 'CLASS_PLACEHOLDER', func_code, count=1)
                    # Replace the placeholder with an empty string to effectively remove the class name
                    func_code = func_code.replace('CLASS_PLACEHOLDER', '')

                # Replace the standalone function name
                pattern = r'\b' + re.escape(function_name) + r'\b\s*\('
                func_code = re.sub(pattern, 'unknown_func(', func_code, count=1)

                return func_code

            # filter the docstring
            def process_docstring(docstring):
                # Find the position of the first period
                first_period_index = docstring.find('.')
                
                # If no period is found, return the entire docstring
                if first_period_index == -1:
                    summary = docstring.strip()
                else:
                    # Extract the content before the first period
                    summary = docstring[:first_period_index + 1].strip()
                
                # If the extracted content is empty, return a default message
                if not summary:
                    return "No summary available."
                
                # Convert the first letter to lowercase
                summary = summary[0].lower() + summary[1:]

                # Remove newlines and replace them with spaces
                summary = summary.replace('\n', ' ').replace('\r', ' ')

                # Replace {@link @code ClassName Alias} with ClassName
                summary = re.sub(r'\{@link\s+([^\s}]+)\s*[^}]*\}', r'\1', summary)
                summary = re.sub(r'\{@code\s+([^\s}]+)\s*[^}]*\}', r'\1', summary)

                # Replace incomplete @link @code tags with their content
                summary = re.sub(r'\{@link\s+([^\s}]+)\s*[^}]*', r'\1', summary)
                summary = re.sub(r'\{@code\s+([^\s}]+)\s*[^}]*', r'\1', summary)

                return summary

            # delete the class_name of func_name
            def process_function_name(func_name):
                # Extract the class name and function name from func_name
                if '.' in func_name:
                    class_name, function_name = func_name.split('.')
                else:
                    class_name = None
                    function_name = func_name

                return function_name


            source = [prompt.format(code=replace_function_name(func_code, func_name)) for func_code, func_name in zip(examples["code"], examples["func_name"])]
            target = ["The function " + process_docstring(docstring) + "\nSuggested Function Name: " + process_function_name(func_name) for docstring, func_name in zip(examples["docstring"], examples["func_name"])]

            model_inputs = tokenizer(source, max_length=args.max_source_len, padding="max_length", truncation=True)
            labels = tokenizer(target, max_length=args.max_target_len, padding="max_length", truncation=True)
            
            model_inputs["labels"] = labels["input_ids"].copy()
            model_inputs["labels"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"]
            ]
            return model_inputs

        train_data = datasets["train"].map(
            preprocess_function,
            batched=True,
            remove_columns=datasets["train"].column_names,
            num_proc=64,
            load_from_cache_file=False,
        )
        print(f'  ==> Loaded {len(train_data)} samples')
        train_data.save_to_disk(args.cache_data)
        print(f'  ==> Saved to {args.cache_data}')
        return train_data



def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    # Save command to file
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    # Load and tokenize data using the tokenizer from `args.load`. If the data is already cached, load it from there.
    # You can customize this function to load your own data for any Seq2Seq LM tasks.
    train_data = load_tokenize_data(args)

    if args.data_num != -1:
        train_data = train_data.select([i for i in range(args.data_num)])

    # Load model from `args.load`
    model = AutoModelForSeq2SeqLM.from_pretrained(args.load)
    print(f"  ==> Loaded model from {args.load}, model size {model.num_parameters()}")

    run_training(args, model, train_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeT5 finetuning on Seq2Seq LM task")
    parser.add_argument('--data-num', default=-1, type=int)
    parser.add_argument('--max-source-len', default=512, type=int)
    parser.add_argument('--max-target-len', default=128, type=int)
    parser.add_argument('--instruct-data-path', default='./datasets/CodeSearchNet_Java/java/final/jsonl', type=str)
    parser.add_argument('--cache-data', default='cache_data/summarize_codet5', type=str)
    parser.add_argument('--load', default='./codet5-base-multi-sum', type=str)

    # Training
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr-warmup-steps', default=200, type=int)
    parser.add_argument('--batch-size-per-replica', default=8, type=int)
    parser.add_argument('--grad-acc-steps', default=4, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--deepspeed', default=None, type=str)
    parser.add_argument('--fp16', default=True, action='store_true')

    # Logging and stuff
    parser.add_argument('--save-dir', default="./my_instruct_codet5_sum_3epoch", type=str)
    parser.add_argument('--log-freq', default=10, type=int)
    parser.add_argument('--save-freq', default=500, type=int)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)