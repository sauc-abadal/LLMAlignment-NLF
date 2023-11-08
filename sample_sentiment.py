import torch
from pathlib import Path
import os
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from statistics import mean
import pandas as pd
from tqdm import tqdm
from policy import Policy
from torch.utils.data import DataLoader
from main import PromptDataset, PromptCollator
from reward import collate
from utils.utils import load_jsonl, ensure_dir, reduce_sum
from utils.perspective_api import PerspectiveWorker, make_generations_col
from utils.utils import batchify
from torch.nn import functional as F
import json

def expand(tensor, num_repeat):
    return torch.reshape(tensor[:, None].expand(-1, num_repeat, -1), [batch_size * num_repeat, -1])


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def distinctness(responses, num_sample):
    generations_batch = list(chunks(responses, num_sample))
    dist1, dist2, dist3 = [], [], []
    # calculate dist1, dist2, dist3 across generations for every prompt
    for generations in tqdm(generations_batch, total=len(generations_batch), desc='Evaluating diversity'):
        unigrams, bigrams, trigrams = set(), set(), set()
        total_words = 0
        for gen in generations:
            o = gen.split(' ')
            total_words += len(o)
            unigrams.update(o)
            for i in range(len(o) - 1):
                bigrams.add(o[i] + '_' + o[i + 1])
            for i in range(len(o) - 2):
                trigrams.add(o[i] + '_' + o[i + 1] + '_' + o[i + 2])
        dist1.append(len(unigrams) / total_words)
        dist2.append(len(bigrams) / total_words)
        dist3.append(len(trigrams) / total_words)

    # take the mean across prompts
    return np.nanmean(dist1), np.nanmean(dist2), np.nanmean(dist3)

if __name__ == "__main__":
    print("Script started!")
    parser = argparse.ArgumentParser(description='Script for evaluation: Quark generalization to Sentiment experiment')
    parser.add_argument('--save_path', type=str, default='output/savepath', help='Path for saving evaluation results')
    parser.add_argument('--checkpoint_path', type=str, default='/cluster/work/sachan/sauc/quark/ckp_11000.pth', help='Path to the model under evaluation checkpoint, set this to None to evaluate a baseline model')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for data processing')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top p value')
    parser.add_argument('--num_samples', type=int, default=25, help='Number of generations for each test prompt')
    parser.add_argument('--test_set_path_negative', type=str, default='data/sentiment/sentiment_prompts-10k/negative_prompts.jsonl', help='Path to the test set containing negative prompts')
    parser.add_argument('--test_set_path_positive', type=str, default='data/sentiment/sentiment_prompts-10k/positive_prompts.jsonl', help='Path to the test set containing positive prompts')
    parser.add_argument('--test_set_path_neutral', type=str, default='data/sentiment/sentiment_prompts-10k/neutral_prompts.jsonl', help='Path to the test set containing neutral prompts')
    parser.add_argument('--positive_experiment', action="store_true", default=False, help='Whether to evaluate on steering away from Negative/Positive sentiment. If specifyied, steering away from Positive')

    args = parser.parse_args()

    save_path = args.save_path
    checkpoint_path = args.checkpoint_path
    batch_size = args.batch_size
    top_p = args.top_p
    num_samples = args.num_samples
    test_set_path_negative = args.test_set_path_negative
    test_set_path_positive = args.test_set_path_positive
    test_set_path_neutral = args.test_set_path_neutral
    positive_experiment = args.positive_experiment
    n_extra_tokens = 5
    model = 'gpt2-large'
    print(f"Checkpoint path to be loaded: {checkpoint_path}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ensure_dir(save_path)

    tree_tokens = [' _TREE_TOKEN_{}'.format(str(idx).zfill(5)) for idx in range(n_extra_tokens)] + \
              [' _TREE_TOKEN_ZERO_COMMENTS']
    print(f"Using {n_extra_tokens} quantiles, associated with the following tokens {tree_tokens}")

    policy = Policy(model_name=model, temperature=1.0, device=device, reward_cond=True, tree_tokens=tree_tokens)
    best_cat_id = policy.tokenizer.convert_tokens_to_ids(tree_tokens[0])
    prompt_collator = PromptCollator(tokenizer=policy.tokenizer)

    if positive_experiment:
        print("Steering away from Positive sentiment")
    else:
        print("Steering away from Negative sentiment")

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        policy.model.load_state_dict(checkpoint['policy_model'])

    reward_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    reward_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    reward_model.eval()
    reward_model = reward_model.to(device)

    print('Model initialization done!')

    def process_dataset(dataset_path, save_path, batch_size, num_samples, top_p):
        test_dataset = PromptDataset(path=dataset_path)
        dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=prompt_collator, drop_last=True)
        print(f"Loading {dataset_path}")
        print(f"The dataloader has {len(dataloader)} batches, which will lead to a total of {len(dataloader)*batch_size*num_samples} generations with associated sentiment score.")
        reward_file = Path(save_path) / f'reward_{dataset_path.split("/")[-1].split(".")[0]}.json'
        print(f"Saving results to {reward_file}")

        responses, prompts = [], []
        for i, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
            input_ids, attention_mask = batch

            input_ids = torch.cat([input_ids.new([best_cat_id] * len(input_ids))[:, None], input_ids], dim=1)
            attention_mask = torch.cat([attention_mask.new([1] * len(attention_mask))[:, None], attention_mask], dim=1)

            outputs = policy.sample(input_ids=expand(input_ids, num_samples), attention_mask=expand(attention_mask, num_samples),
                                    top_p=top_p)

            prompt, response = outputs['query/text'], outputs['response/text']
            prompts.extend([x for n, x in enumerate(prompt) if not n % num_samples])
            responses.extend(response)

            pbar = tqdm(total=len(response), dynamic_ncols=True)
            pbar.set_description(f'Computing Positivity-Sentiment scores')
            for batch_prompts, batch_responses in zip(batchify(prompt, num_samples), batchify(response, num_samples)):
                with torch.no_grad():
                    input_dict = reward_tokenizer(batch_responses, padding=True, return_tensors="pt")
                    input_ids = input_dict["input_ids"].to(device)
                    attention_mask = input_dict["attention_mask"].to(device)
                    logits = reward_model(input_ids, attention_mask).logits
                    positivity_scores = F.softmax(logits, dim=1)[:, 1]

                    response_dict = {
                        'prompt': batch_prompts[0],  # Taking the first element assuming they are all the same
                        'generations': []
                    }
                    for idx in range(len(batch_responses)):
                        generation_dict = {
                            'text': batch_responses[idx],
                            'positive_score': positivity_scores[idx].item()
                        }
                        response_dict['generations'].append(generation_dict)

                    # Storing each dictionary in a JSONL format
                    with open(reward_file, 'a') as f:
                        json.dump(response_dict, f)
                        f.write('\n')
                    
                    pbar.update(len(batch_responses))
        
        assert os.path.exists(reward_file), 'missing reward file'
        # Load the JSONL file into a list of dictionaries
        with open(reward_file, 'r') as file:
            results = [json.loads(line) for line in file]

        # Compute the mean percentage of positive scores > 0.5 for each prompt
        mean_percentages = []
        for prompt_data in results:
            positive_scores = [gen['positive_score'] for gen in prompt_data['generations'] if gen['positive_score'] is not None]
            if positive_scores:
                # Calculate the percentage of positive scores > 0.5
                positive_over_half = sum(score > 0.5 for score in positive_scores)
                mean_percentage = (positive_over_half / len(positive_scores)) * 100
                mean_percentages.append(mean_percentage)

        # Calculate the overall mean percentage
        overall_mean_percentage = sum(mean_percentages) / len(mean_percentages) if mean_percentages else 0
        print(f'Mean percentage of positive continuations among the {num_samples} generations for the {dataset_path.split("/")[-1].split(".")[0]}: {overall_mean_percentage}')

        return responses, overall_mean_percentage


    if positive_experiment:
        responses, mean_percentage_positive = process_dataset(dataset_path=test_set_path_positive, save_path=save_path, batch_size=batch_size, num_samples=num_samples, top_p=top_p)
    else:
        responses, mean_percentage_negative = process_dataset(dataset_path=test_set_path_negative, save_path=save_path, batch_size=batch_size, num_samples=num_samples, top_p=top_p)
    
    responses_neutral, mean_percentage_neutral = process_dataset(dataset_path=test_set_path_neutral, save_path=save_path, batch_size=batch_size, num_samples=num_samples, top_p=top_p)
    
    dist1, dist2, dist3 = distinctness(responses + responses_neutral, num_samples)
    print(f'dist-1={dist1:.3f}, dist-2={dist2:.3f}, dist-3={dist3:.3f}')

    # write output results
    with open(f'{save_path}/eval_results.txt', 'w') as fo:

        if positive_experiment:
            fo.write(f'% Positive (positive prompts) = {mean_percentage_positive:.3f}\n')
        else:
            fo.write(f'% Positive (negative prompts) = {mean_percentage_negative:.3f}\n')

        fo.write(f'% Positive (neutral prompts) = {mean_percentage_neutral:.3f}\n')
        fo.write(f'dist-1={dist1:.3f}, dist-2={dist2:.3f}, dist-3={dist3:.3f}\n')

