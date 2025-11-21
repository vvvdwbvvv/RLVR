import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

from data_types import MiniBatch
from tokenizer import Tokenizer


SYSTEM_MESSAGE = (
    "You are a professional medical assistant. Your task is to generate the 'Assessment' and 'Plan' sections "
    "of a SOAP note based on the 'Subjective' and 'Objective' information provided, along with relevant context from retrieved documents. "
    "First, think through the diagnosis and treatment plan, then provide the answer in the specified format."
)

USER_TEMPLATE = (
    "Based on the following clinical information, please generate the Assessment and Plan.\n\n"
    "## Retrieved Documents:\n"
    "{retrieved_docs}\n\n"
    "## Subjective (S):\n"
    "{s_part}\n\n"
    "## Objective (O):\n"
    "{o_part}\n\n"
    "Generate the 'Assessment' and 'Plan' sections. Show your reasoning process in <think> tags, "
    "and provide the final output in <answer> tags with 'Assessment:' and 'Plan:' headings."
)

RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"
ANSWER_PROMPT = "</think>\n<answer>\n## Assessment:\n" #

class EHRSOAPDataset(Dataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        data_path: str,
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        file_path = Path(data_path) / f"{split}.jsonl"
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
        
        self.full_response_prompt = RESPONSE_PROMPT + ANSWER_PROMPT

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        s_part = item.get('soap', {}).get('S', '')
        o_part = item.get('soap', {}).get('O', '')
        a_part = item.get('soap', {}).get('A', '')
        p_part = item.get('soap', {}).get('P', '')
        retrieved_docs = item.get('retrieved_docs', [])
        formatted_docs = "\n".join([f"- {doc['text']}" for doc in retrieved_docs])
        prefix_data = self.encode_prefix(s_part, o_part, formatted_docs)
        item.update(prefix_data)

        item['target_text'] = f"## Assessment:\n{a_part}\n## Plan:\n{p_part}"
        
        return item

    def encode_prefix(self, s_part: str, o_part: str, retrieved_docs: str):
        user_message = USER_TEMPLATE.format(
            retrieved_docs=retrieved_docs,
            s_part=s_part,
            o_part=o_part
        )
        
        prefix = self.tokenizer.encode_chat_with_response_prompt(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            self.full_response_prompt,
        )
        tokens = self.tokenizer.tokenize(prefix)
        return {
            "prefix": prefix,
            "prefix_tokens": tokens.tokens,
            "prefix_token_ids": tokens.ids,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> MiniBatch:
        prefix_token_ids = [item["prefix_token_ids"] for item in batch]
        
        return MiniBatch(
            items=batch,
            prefix_token_ids=prefix_token_ids,
        )


def reward_function(
    response: str,
    end_token: Optional[str] = None,
    **kwargs, 
) -> Dict[str, Any]:
    """Calculate reward for the generated response."""
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]

    full_response = "<think>" + response 
    
    reward = 0.0
    
    if "Assessment:" in full_response:
        reward += 0.5
        
    if "Plan:" in full_response:
        reward += 0.5
        
    
    return {
        "reward": reward,
        "reward_info": {
            "has_assessment": "Assessment:" in full_response,
            "has_plan": "Plan:" in full_response,
        },
    }