import json
from pathlib import Path
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from typing import Optional, List, Iterable, Dict, Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils.utils import batchify, load_jsonl

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


class Reward:
    def __init__(self, save_path: str, model_name_or_path: str, batch_size: int, device: str):
        self.path = save_path
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path)
        self.model.eval()
        self.model = self.model.to(self.device)

    def get_reward(self, prompts: List[str], responses: List[str], epoch: str) -> List[float]:
        reward_file = Path(self.path) / f'sentiment_{epoch}.json'
        assert len(prompts) == len(responses), f'prompts({len(prompts)}) and responses({len(responses)}) mismatch'


        pbar = tqdm(total=len(prompts), dynamic_ncols=True)
        pbar.set_description(f'Computing Positivity-Sentiment scores')

        rewards = []

        with reward_file.open('a') as f:
            for batch_prompts, batch_responses in zip(batchify(prompts, self.batch_size), batchify(responses, self.batch_size)):
                with torch.no_grad():
                    input_dict = self.tokenizer(batch_responses, padding=True, return_tensors="pt")
                    input_ids = input_dict["input_ids"].to(self.device)
                    attention_mask = input_dict["attention_mask"].to(self.device)
                    logits = self.model(input_ids, attention_mask).logits
                    positivity_scores = F.softmax(logits, dim=1)[:, 1]

                    for idx in range(len(batch_responses)):
                        response_dict = {
                            'prompt': batch_prompts[idx],
                            'generations': {
                                'text': batch_responses[idx],
                                'positive_score': positivity_scores[idx].item()
                            }
                        }
                        json.dump(response_dict, f)
                        f.write('\n')

                        rewards.append(positivity_scores[idx].item())
                    
                    pbar.update(len(batch_responses))

        assert os.path.exists(reward_file), 'missing reward file'

        return rewards

