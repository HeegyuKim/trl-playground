from typing import Optional, List, Iterable, Dict, Any

import torch
from torch import FloatTensor, LongTensor
from tqdm import tqdm
import os
import jsonlines
from dataclasses import dataclass

from transformers import (
    pipeline, LogitsProcessor, RobertaTokenizer, RobertaForSequenceClassification,
    LogitsProcessorList, LlamaForCausalLM, LlamaTokenizer, AutoModelForSequenceClassification, AutoTokenizer
)

import jsonlines
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import LongTensor, FloatTensor

class LlamaForCausalLMForGate(LlamaForCausalLM):
    
    def __init__(self, config):
        super().__init__(config)
        self.ctg_warper = None

    def _get_logits_warper(self, *args, **kwargs) -> LogitsProcessorList:
        warpers = super()._get_logits_warper(*args, **kwargs)
        if self.ctg_warper is not None:
            self.ctg_warper.original_warpers = warpers
            return self.ctg_warper
        else:
            return warpers

class GatedLogitsProcessor(LogitsProcessor):
    def __init__(self, theta, generation_tokenizer, classifier_tokenizer, classifier, post_processor: LogitsProcessor, label_index: int = 1) -> None:
        super().__init__()
        self.theta = theta
        self.post_processor = post_processor
        self.original_warpers = None
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
        warped_scores = self.original_warpers(input_ids, scores)
        next_tokens = torch.multinomial(warped_scores.softmax(-1), num_samples=1) # (b, 1)
        current_ids = torch.cat([input_ids, next_tokens], 1) # (b, s + 1)
        current_texts = self.generation_tokenizer.batch_decode(current_ids, skip_special_tokens=True)
        gate_output = self._classify_text(current_texts)
        

        logits = torch.ones_like(scores, device=scores.device) * -50000
        logits[torch.range(0, logits.shape[0] - 1, device=scores.device, dtype=torch.long), next_tokens.long().squeeze(1)] = 0

        if gate_output.sum().item() > 0:
            print(current_texts)
            print("toxic!", gate_output.cpu())
            gate_output = gate_output.unsqueeze(1)

            guided = self.post_processor(input_ids, scores).to(logits.device)
            gated_scores = logits * (1 - gate_output) + gate_output * guided
            return self.original_warpers(input_ids, gated_scores)
        else:
            return logits
        
class DExpertLogitsProcessor(LogitsProcessor):

    def __init__(self, alpha, expert) -> None:
        super().__init__()
        self.alpha = alpha
        self.expert = expert

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor: # of shape (batch_size, config.vocab_size)
        expert_logits = self.expert(input_ids.to(self.expert.device)).logits[:, -1]
        # return scores + self.alpha * expert_logits.to(scores.device)
        return expert_logits.to(scores.device)    


@dataclass
class DExpertGenerator:
    model_name: str
    expert_model_name: str
    num_return_sequences: int
    max_tokens: int
    p: float = 1.0
    alpha: float = 1.0
    theta: float = 0.5
    classifier_model_name: Optional[str] = None
    device1: str = "cuda:0"
    device2: str = "cuda:1"

    def __post_init__(self):
        self.model = LlamaForCausalLMForGate.from_pretrained(self.model_name).to(self.device1).eval()
        self.expert = LlamaForCausalLM.from_pretrained(self.expert_model_name).to(self.device2).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        processor = DExpertLogitsProcessor(
            alpha=self.alpha,
            expert=self.expert,
        )
        if self.classifier_model_name and self.classifier_model_name != "no":
            processor = GatedLogitsProcessor(
                self.theta,
                self.tokenizer,
                AutoTokenizer.from_pretrained(self.classifier_model_name),
                AutoModelForSequenceClassification.from_pretrained(self.classifier_model_name).to(self.device1).eval(),
                processor
            )

        # self.logits_processors = LogitsProcessorList([processor])
        self.model.ctg_warper = processor
    
    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device1)

        outputs = self.model.generate(
            **inputs,
            num_return_sequences=self.num_return_sequences,
            max_new_tokens=self.max_tokens,
            top_p=self.p,
            do_sample=True,
            early_stopping=True,
            pad_token_id=50256
        )
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outputs = [x[len(prompt):] for x in outputs]
        return outputs
