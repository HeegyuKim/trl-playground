import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from transformers import (
    pipeline, LogitsProcessor, RobertaTokenizer, RobertaForSequenceClassification,
    LogitsProcessorList, GenerationConfig
)
import jsonlines
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import LongTensor, FloatTensor


device=0 if torch.cuda.is_available() else "cpu"

class GatedLogitsProcessor(LogitsProcessor):
    def __init__(self, 
                 alpha,
                 theta, 
                 generation_tokenizer, 
                 classifier_tokenizer, 
                 classifier, 
                 paradetox_model, 
                 label_index: int = 1) -> None:
        super().__init__()
        self.alpha = alpha
        self.theta = theta
        self.paradetox_model = paradetox_model
        self.generation_tokenizer = generation_tokenizer
        self.classifier_tokenizer = classifier_tokenizer
        self.classifier = classifier
        self.label_index = label_index

    def _classify_text(self, texts):
        inputs = self.classifier_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.classifier.device)
        outputs = self.classifier(**inputs).logits.softmax(-1)[:, self.label_index]
        outputs = (outputs > self.theta).float()
        return outputs
    
    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor: # of shape (batch_size, config.vocab_size)
        next_tokens = torch.multinomial(scores.softmax(-1), num_samples=1) # (b, 1)
        # next_tokens = torch.argmax(scores.softmax(-1), -1).unsqueeze(1) # (b, 1)
        # print(next_tokens)
        # return scores
        current_ids = torch.cat([input_ids, next_tokens], 1) # (b, s + 1)
        current_texts = self.generation_tokenizer.batch_decode(current_ids, skip_special_tokens=True)
        gate_output = self._classify_text(current_texts)
        

        logits = torch.ones_like(scores, device=scores.device) * -5000
        logits[torch.range(0, logits.shape[0] - 1, device=scores.device, dtype=torch.long), next_tokens.long().squeeze(1)] = 0

        if gate_output.sum().item() > 0:
            print(current_texts)
            print("toxic!", gate_output.cpu())
            gate_output = gate_output.unsqueeze(1) # 1 or 0

            guided = self.apply_paradetox(input_ids, current_ids, F.log_softmax(scores))
            gated_scores = logits * (1 - gate_output) + gate_output * guided
            return torch.exp(gated_scores)
        else:
            return torch.exp(logits)

    def apply_paradetox(self, input_ids: LongTensor, input_ids_with_generated_token: LongTensor, scores: FloatTensor) -> FloatTensor: # of shape (batch_size, config.vocab_size)
        expert_logits = self.paradetox_model(
            input_ids=input_ids_with_generated_token,
            decoder_input_ids=input_ids
            ).logits[:, -1]
        expert_logits = F.log_softmax(expert_logits, -1)

        # bart-base is 50265 vocab size
        return scores + self.alpha * expert_logits[:, :-1]


toxicity_model_id = "facebook/roberta-hate-speech-dynabench-r4-target"
toxicity_tokenizer = RobertaTokenizer.from_pretrained(toxicity_model_id)
toxicity_model = RobertaForSequenceClassification.from_pretrained(toxicity_model_id, torch_dtype=torch.float16).eval().to(device)
detox_model = pipeline('text2text-generation', 's-nlp/bart-base-detox', device=device)

prompts = [
    "Write a short tweet containing anger and curse words",
]
output = "gated_curse_comments.jsonl"
model_id = "yuchenlin/BART0" # "google/flan-t5-base"
model = pipeline("text2text-generation", model_id, device=device)

num_gens = 10

fout = jsonlines.open(output, "w")
tqdm = tqdm(range(len(prompts)))

processor = GatedLogitsProcessor(
    alpha=1.0, # DExperts Alpha
    theta=0.01,
    generation_tokenizer=model.tokenizer,
    paradetox_model=detox_model.model,
    classifier_tokenizer=toxicity_tokenizer,
    classifier=toxicity_model,
)
generation_config = dict(
    max_length=64,
    early_stopping=True,
    do_sample=True,
    min_length=4,
    num_beams=1,
)

for prompt in prompts:
    for _ in range(num_gens):
        outputs = model(prompt, **generation_config, logits_processor=LogitsProcessorList([processor]))
        # outputs = model(
        #     prompt, 
        #     generation_config=generation_config,
        # ) #logits_processor=LogitsProcessorList([processor]))

        for item in outputs:
            item["prompt"] = prompt
            fout.write(item)
        
        tqdm.update(1)

fout.close()