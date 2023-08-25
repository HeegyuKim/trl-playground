import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from transformers import pipeline
import jsonlines
from tqdm import tqdm
import torch


prompts = [
    "Please write a positive restaurant review at least 2 sentences long. Don't make any mistakes in grammar.",
    "Please write a negative restaurant review at least 2 sentences long. Don't make any mistakes in grammar.",
]
output = "flan_sentimental_review_baseline.jsonl"
model_id = "google/flan-t5-large"
model = pipeline("text2text-generation", model_id, device=0 if torch.cuda.is_available() else "cpu")

generation_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "temperature": 1.0,
    "num_return_sequences": 8,
    "max_new_tokens": 128,
    "min_length": 16,
    "do_sample": True,
    "early_stopping": True
}

num_gens = 10000

fout = jsonlines.open(output, "w")
prompt_iter_count = num_gens // generation_kwargs["num_return_sequences"]
tqdm = tqdm(range(prompt_iter_count * len(prompts)))

for prompt in prompts:
    for _ in range(prompt_iter_count):
        outputs = model(prompt, **generation_kwargs)

        for item in outputs:
            item["prompt"] = prompt
            fout.write(item)
        
        tqdm.update(1)

fout.close()