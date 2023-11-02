from datasets import load_dataset

dataset = load_dataset("euclaise/writingprompts", split="test")
print(dataset)
print(dataset.features)
dataset = dataset.remove_columns("story")
print(dataset)

def processing_function(example):
    idx = example["prompt"].find("]")
    
    if idx > 10:
        example["prompt"] = {"text": example["prompt"][:-7]}
    elif idx != -1:
        example["prompt"] = {"text": example["prompt"][idx+2:]}
    else:
        example["prompt"] = {"text": example["prompt"]}
    return example

dataset = dataset.map(processing_function, batched=False)
print(dataset)
print(dataset[:5])

savepath = "data/toxicity/out_of_domain/test.jsonl"
dataset.to_json(savepath)
