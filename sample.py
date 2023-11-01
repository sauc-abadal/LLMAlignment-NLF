import argparse
import torch
from pathlib import Path
import os
import numpy as np
import json

from statistics import mean
from tqdm import tqdm
from policy import Policy
from torch.utils.data import DataLoader
from datasets_and_collators import PromptDataset, PromptCollator
from utils.utils import load_jsonl, ensure_dir, reduce_sum
from reward import Reward
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn import functional as F
from utils.utils import batchify

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

def add_best_control_code(input_ids, attention_mask, reward_cond=True):
        """
        Prepend control tokens associated with the best performing quantile to a batch of input sequences.

        This function takes a batch of input token IDs and their corresponding attention masks and adds control tokens
        associated with the best performing quantile to the beginning of each input sequence. It also inserts a special
        <|separator|> token between the control tokens and the original input tokens (newly added as not contemplated within the GPT2Tokenizer).

        Args:
            self (object): The instance of the class containing this method.
            input_ids (torch.Tensor): A tensor containing token IDs for a batch of input sequences.
            attention_mask (torch.Tensor): A tensor containing attention masks for the input sequences.

        Returns:
            torch.Tensor: A tensor containing the modified input token IDs with control tokens prepended, and the separator token.
            torch.Tensor: A tensor containing the modified attention masks.

        Note:
            - `self.best_cat_ids` should be set to the control tokens associated with the best performing quantile.
            - The <|separator|> token is used to separate the control tokens from the input tokens.
        """
        if not reward_cond:
            return input_ids, attention_mask
        
        input_ids = torch.cat([input_ids.new([best_cat_id] * len(input_ids)),
                            input_ids.new([[policy.tokenizer.sep_token_id]]*len(input_ids)),
                                input_ids], dim=1)
        
        attention_mask = torch.cat([attention_mask.new([[1]*len(best_cat_id)] * len(attention_mask)), 
                                    attention_mask.new([[1]]*len(attention_mask)),
                                    attention_mask], dim=1)

        return input_ids, attention_mask

def remove_any_control_code(input_ids, attention_mask, rmv_sep_token=False, reward_cond=True):
        """
        Remove control tokens from a batch of input sequences.

        This function takes a batch of input token IDs and their corresponding attention masks and removes control tokens
        added for conditioning during generation. It also provides the option to remove the separator token.

        Args:
            self (object): The instance of the class containing this method.
            input_ids (torch.Tensor]): A tensor containing token IDs for a batch of input sequences.
            attention_mask (torch.Tensor]): A tensor containing attention masks for the input sequences.
            rmv_sep_token (bool, optional): Set to True to remove the separator token from the sequences.

        Returns:
            torch.Tensor]: A tensor containing the modified input token IDs with control tokens removed.
            torch.Tensor]: A tensor containing the modified attention masks.

        Note:
            - Control tokens are removed from each sequence, and the separator token can also be removed if specified.
        """
        if not reward_cond:
            return input_ids, attention_mask

        bs, _ = input_ids.shape

        sep_token_id = policy.tokenizer.sep_token_id
        sep_token_mask = (input_ids == sep_token_id)
        cumulative_mask = sep_token_mask.cumsum(dim=1)
        tokens_after_special_mask = cumulative_mask > 0
        
        input_ids = input_ids[tokens_after_special_mask].reshape(bs, -1)
        attention_mask = attention_mask[tokens_after_special_mask].reshape(bs, -1)

        if rmv_sep_token:
            input_ids = input_ids[:, 1:]
            attention_mask = attention_mask[:, 1:]
            
        return input_ids, attention_mask
    
# MODIFIED
def decode(query_input_ids, response_input_ids=None, skip_special_tokens=True):
    """
    Decode token sequences into human-readable text.

    This function takes token IDs or sequences and converts them into human-readable text using the tokenizer's decoding
    capabilities.

    Args:
        self (object): The instance of the class containing this method.
        query_input_ids (torch.Tensor or List[List[int]]): A tensor or list of token IDs representing input sequences.
        response_input_ids (torch.Tensor or List[List[int]], optional): A tensor or list of token IDs representing response
            sequences. If not provided (None), only the input sequences are decoded.

    Returns:
        List[str] or Tuple[List[str], List[str]]: If `response_input_ids` is provided, it returns a tuple containing two lists:
        1. List of decoded input sequences.
        2. List of decoded response sequences.
        If `response_input_ids` is not provided, it returns a list containing the decoded input sequences.
    """

    query = [policy.tokenizer.decode(p, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=True)
            for p in query_input_ids]
        
    if response_input_ids is None:
        return query

    response = [policy.tokenizer.decode(r, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for r in response_input_ids]
    return query, response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for evaluation: toxicity and distinctness')
    parser.add_argument('--save_path', type=str, default='output/savepath', help='Path for saving evaluation results')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to the model under evaluation checkpoint, set this to None to evaluate a baseline model')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for data processing')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top p value')
    parser.add_argument('--num_samples', type=int, default=25, help='Number of generations for each test prompt')
    parser.add_argument('--test_set_path_negative', type=str, default='data/sentiment/raw/sentiment_prompts-10k/negative_prompts.jsonl', help='Path to the test set containing negative prompts')
    parser.add_argument('--test_set_path_neutral', type=str, default='data/sentiment/raw/sentiment_prompts-10k/neutral_prompts.jsonl', help='Path to the test set containing neutral prompts')
    parser.add_argument('--rate_limit', type=int, default=120, help='PerspectiveAPI Rate limit value')
    parser.add_argument('--reward_cond', type=bool, default=True, help='Wether to use NLF reward tokens or not, set this to False to evaluate a baseline model')

    args = parser.parse_args()

    save_path = args.save_path
    checkpoint_path = args.checkpoint_path
    batch_size = args.batch_size
    top_p = args.top_p
    num_samples = args.num_samples
    test_set_path_negative = args.test_set_path_negative
    test_set_path_neutral = args.test_set_path_neutral
    rate_limit = args.rate_limit
    reward_cond = args.reward_cond
    model = 'gpt2-large'
    print(f"Checkpoint path to be loaded: {checkpoint_path}")
    print(f"Using NLF reward tokens: {reward_cond}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ensure_dir(save_path)

    policy = Policy(model_name=model, temperature=1.0, device=device, reward_cond=reward_cond)
    prompt_collator = PromptCollator(tokenizer=policy.tokenizer)

    tags = ["Lowest Negativity", "Low-Moderate Negativity", "Moderate Negativity", "High-Moderate Negativity", "Maximum Negativity"]
    tree_tokens = [policy.tokenizer.convert_ids_to_tokens(policy.tokenizer(tag)["input_ids"]) for tag in tags]
    best_cat_id = policy.tokenizer.convert_tokens_to_ids(tree_tokens[0])

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        policy.model.load_state_dict(checkpoint['policy_model'])

    reward = Reward(save_path=save_path, 
                    model_name_or_path="distilbert-base-uncased-finetuned-sst-2-english", 
                    batch_size=batch_size*num_samples,
                    deivce=device
                    )
    reward_file = Path(save_path) / 'reward.json'
    reward_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    reward_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    reward_model.eval()
    reward_model = reward_model.to(device)

    print('Model initialization done!')
    
    def process_dataset(dataset_path, reward_file, batch_size, num_samples, top_p, reward_cond):

        test_dataset = PromptDataset(path=dataset_path)
        dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=prompt_collator, drop_last=True)
        print(f"Loading {dataset_path}")
        print(f"The dataloader has {len(dataloader)} batches, which will lead to a total of {len(dataloader)*batch_size*num_samples} generations with associated toxicity score.")
        
        responses = []
        for i, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
            input_ids, attention_mask = batch
            input_ids, attention_mask = add_best_control_code(input_ids, attention_mask, reward_cond=reward_cond)
            outputs = policy.sample(input_ids=expand(input_ids, num_samples), attention_mask=expand(attention_mask, num_samples),
                                    top_p=top_p)

            input_ids, _ = remove_any_control_code(input_ids, attention_mask, rmv_sep_token=True, reward_cond=reward_cond)
            prompt = decode(input_ids)
            response = outputs['response/text']

            responses.extend(response)

            pbar = tqdm(total=len(response), dynamic_ncols=True)
            pbar.set_description(f'Computing Positivity-Sentiment scores')
            for batch_prompts, batch_responses in zip(batchify([p for p in prompt for _ in range(num_samples)], num_samples), batchify(response, num_samples)):
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
        print(f"Mean percentage of positive continuations among the {num_samples} generations for each prompt: {overall_mean_percentage}")

        return responses, overall_mean_percentage


    responses_negative, mean_percentage_negative = process_dataset(dataset_path=test_set_path_negative, reward_file=reward_file,
                                                                   batch_size=batch_size, num_samples=num_samples, top_p=top_p, reward_cond=reward_cond)
    responses_neutral, mean_percentage_neutral = process_dataset(dataset_path=test_set_path_neutral, reward_file=reward_file,
                                                                   batch_size=batch_size, num_samples=num_samples, top_p=top_p, reward_cond=reward_cond)
    dist1, dist2, dist3 = distinctness(responses_negative.extend(responses_neutral), num_samples)
    print(f'dist-1={dist1:.3f}, dist-2={dist2:.3f}, dist-3={dist3:.3f}')

    # write output results
    with open(f'{save_path}/eval_results.txt', 'w') as fo:
        fo.write(f'% Positive (negative prompts) = {mean_percentage_negative:.3f}\n')
        fo.write(f'% Positive (neutral prompts) = {mean_percentage_neutral:.3f}\n')
        fo.write(f'dist-1={dist1:.3f}, dist-2={dist2:.3f}, dist-3={dist3:.3f}\n')