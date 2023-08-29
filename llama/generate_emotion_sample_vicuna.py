import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import pipeline
import jsonlines
from tqdm import tqdm
import torch


emotions = ["anger", "sadness", "joy", "love", "fear", "surprise"]

system_prompt = "### System:\nYou are StableBeluga, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n"

prompts = [
    (x, f"{system_prompt}### User:\nWrite a tweet where you feel like the author is in {x}.\n\n### Assistant:\nSure, here is a comment. ") for x in emotions
]

# model_id = "meta-llama/Llama-2-7b-chat-hf"
# output = "llama2-chat-emotion.jsonl"

model_id = "stabilityai/StableBeluga-7B"
output = "beluga-chat-emotion.jsonl"


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