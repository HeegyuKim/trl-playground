import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"

from transformers import pipeline
from gated_llama import DExpertGenerator
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
output = "gated-beluga-chat-emotion.jsonl"

expert_model_id = "meta-llama/Llama-2-7b-chat-hf"
toxicity_model_id = "facebook/roberta-hate-speech-dynabench-r4-target"

num_return_sequences=1
generator = DExpertGenerator(
    model_id,
    expert_model_id,
    num_return_sequences,
    64,
    classifier_model_name=toxicity_model_id
)
num_gens = 1000

fout = jsonlines.open(output, "w")
prompt_iter_count = num_gens // num_return_sequences
tqdm = tqdm(range(prompt_iter_count * len(prompts)))

for emotion, prompt in prompts:
    for _ in range(prompt_iter_count):
        outputs = generator.generate(prompt)

        for generated_text in outputs:
            item = {}
            item["generated_text"] = generated_text
            item["prompt"] = prompt
            item["emotion"] = emotion
            fout.write(item)
        
        tqdm.update(1)

fout.close()