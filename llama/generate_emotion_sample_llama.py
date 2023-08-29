import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from transformers import pipeline
import jsonlines
from tqdm import tqdm
import torch


emotions = ["anger", "sadness", "joy", "love", "fear", "surprise"]

prompts = [
    (x, f"[INST] Write a tweet where you feel like the author is in {x}. [/INST] ") for x in emotions
]

model_id = "meta-llama/Llama-2-7b-chat-hf"
output = "llama2-chat-emotion.jsonl"

# model_id = "lmsys/vicuna-7b-v1.5-16k"
# output = "vicuna-chat-emotion.jsonl"


model = pipeline("text-generation", model_id, device=0 if torch.cuda.is_available() else "cpu")
generation_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "temperature": 1.0,
    "num_return_sequences": 1,
    "max_new_tokens": 64,
    "min_length": 8,
    "do_sample": True,
    "early_stopping": True
}

num_gens = 1000

fout = jsonlines.open(output, "w")
prompt_iter_count = num_gens // generation_kwargs["num_return_sequences"]
tqdm = tqdm(range(prompt_iter_count * len(prompts)))

for emotion, prompt in prompts:
    for _ in range(prompt_iter_count):
        outputs = model(prompt, **generation_kwargs)

        for item in outputs:
            item["generated_text"] = item["generated_text"].replace(prompt, "")
            item["prompt"] = prompt
            item["emotion"] = emotion
            fout.write(item)
        
        tqdm.update(1)

fout.close()