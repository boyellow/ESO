import torch
import os
from transformers import GenerationConfig
import torch.nn.functional as F
from rouge_metric import compute_metrics
from rouge_score import rouge_scorer
import numpy as np
import time


class SampleGenerator():
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.max_new_token = self.args.max_length - self.args.max_prompt_length
        self.pad_id = tokenizer.pad_token_id
        self.generation_config = GenerationConfig(
            do_sample=args.do_sample,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_length=args.max_length,
            min_length=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False
        )
        
    def run_sample(self, model, gen_data):
        bs = gen_data["input_ids"].size(0)
        results = {
            "input_ids": torch.ones(bs, self.args.max_length, dtype=torch.long, device=gen_data["input_ids"].device) * self.pad_id,
            "attention_mask": torch.zeros(bs, self.args.max_length, dtype=torch.float,  device=gen_data["input_ids"].device),
            "position_ids": torch.zeros(bs, self.args.max_length, dtype=torch.long,  device=gen_data["input_ids"].device),
            "no_model_batch": torch.ones(bs, self.args.max_length, dtype=torch.long, device=gen_data["input_ids"].device) * -100,
        }
        
        model.eval()
        with torch.no_grad():
            gen_out = model.generate(
                **gen_data,
                generation_config=self.generation_config,
                max_new_tokens=self.max_new_token,
            )
            
            full_ids = gen_out.sequences
            input_ids = full_ids[:, :gen_data["input_ids"].size(1)]
            response_ids = full_ids[:, gen_data["input_ids"].size(1):]
            
            for i in range(len(input_ids)):
                result_id = torch.cat(
                    (input_ids[i][input_ids[i] != self.pad_id],
                     response_ids[i][response_ids[i] != self.pad_id]),
                )
                input_id = input_ids[i][input_ids[i] != self.pad_id]
                response_id = response_ids[i][response_ids[i] != self.pad_id]
                
                results["input_ids"][i, :len(result_id)] = result_id
                results["position_ids"][i, :len(result_id)] = torch.arange(len(result_id))
                results["no_model_batch"][i, len(input_id):len(result_id)] = response_id

        results["attention_mask"] = torch.where(results["input_ids"] != self.pad_id, 1, 0)
        results["attention_mask"] = results["attention_mask"].float()
        results["no_model_batch"] = results["no_model_batch"].long()
        results["label"] = results["no_model_batch"].long()
        return results
    

class NesSampleGenerator():
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id


    def rougel_score(self, preds, targets):
        rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = []
        for pred, target in zip(preds, targets):
            score = rouge.score(prediction=pred, target=target)
            scores.append(score)
        return scores


    def run_sample_temp(self, model, gen_data, model_batch):
        bs = model_batch["input_ids"].size(0)
        
        model.eval()

        with torch.no_grad():
            
            results = []
            temp_list = [0.5, 1.0, 1.25, 1.5]
            max_new_tokens = self.args.max_new_tokens

            target_ids = gen_data.pop("target_ids")

            for it, temp in enumerate(temp_list):

                results_i = {
                    "input_ids": torch.ones(bs, self.args.max_length+max_new_tokens, dtype=torch.long, device=gen_data["input_ids"].device) * self.pad_id,
                    "attention_mask": torch.zeros(bs, self.args.max_length+max_new_tokens, dtype=torch.float,  device=gen_data["input_ids"].device),
                    "position_ids": torch.zeros(bs, self.args.max_length+max_new_tokens, dtype=torch.long,  device=gen_data["input_ids"].device),
                    "no_model_batch": torch.ones(bs, self.args.max_length+max_new_tokens, dtype=torch.long, device=gen_data["input_ids"].device) * -100,
                    "response_ids": torch.ones(bs, self.args.max_length+max_new_tokens, dtype=torch.long, device=gen_data["input_ids"].device) * self.pad_id
                    }

                self.generation_config = GenerationConfig(
                    do_sample=self.args.do_sample,
                    top_p=self.args.top_p,
                    top_k=self.args.top_k,
                    temperature=temp,
                    repetition_penalty=self.args.repetition_penalty,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=False)

                gen_out = model.generate(
                    **gen_data,
                    generation_config=self.generation_config,
                    max_new_tokens=max_new_tokens,
                )
                
                full_ids = gen_out.sequences
                input_ids = full_ids[:, :gen_data["input_ids"].size(1)]
                response_ids = full_ids[:, gen_data["input_ids"].size(1):]
                
                for i in range(len(input_ids)):
                    result_id = torch.cat(
                        (input_ids[i][input_ids[i] != self.pad_id],
                        response_ids[i][response_ids[i] != self.pad_id]),
                    )
                    input_id = input_ids[i][input_ids[i] != self.pad_id]
                    response_id = response_ids[i][response_ids[i] != self.pad_id]

                    results_i["response_ids"][i, :len(response_id)] = response_id
                    
                    results_i["input_ids"][i, :len(result_id)] = result_id
                    results_i["position_ids"][i, :len(result_id)] = torch.arange(len(result_id))
                    results_i["no_model_batch"][i, len(input_id):len(result_id)] = response_id
                
                results_i["attention_mask"] = torch.where(results_i["input_ids"] != self.pad_id, 1, 0)
                results_i["attention_mask"] = results_i["attention_mask"].float()
                results_i["no_model_batch"] = results_i["no_model_batch"].long()

                results.append(results_i)
                
        ref_sentences = [self.tokenizer.decode(ref_id, skip_special_tokens=True).strip() for ref_id in target_ids]

        hyp_sentences_list = []
        for j,hyps in enumerate(results):

            hyp_ids = hyps["response_ids"]
            hyp_senten = [self.tokenizer.decode(hyp_id, skip_special_tokens=True).strip() for hyp_id in hyp_ids]

            hyp_sentences_list.append(hyp_senten)
        
        rouge_scores_dict = [self.rougel_score(hyp_s, ref_sentences) for hyp_s in hyp_sentences_list]

        rouge_scores = []
        for dc in rouge_scores_dict:
            rouge_scores.append(np.array([dc[i]["rougeL"].fmeasure for i in range(len(dc))]).reshape(-1,1))

        rouge_scores = np.concatenate(rouge_scores, axis=1)

        mean_rouge_score = np.mean(rouge_scores, axis=1, keepdims=True)

        diff = torch.tensor(mean_rouge_score - rouge_scores).to(gen_data["input_ids"].device)

        mean_rouge_score = torch.tensor(mean_rouge_score).to(gen_data["input_ids"].device)

        collected_results = {"input_ids":[], "attention_mask":[], "position_ids":[], "labels":[]}

        for res_i in results:
            collected_results["input_ids"].append(res_i["input_ids"])
            collected_results["attention_mask"].append(res_i["attention_mask"])
            collected_results["position_ids"].append(res_i["position_ids"])
            collected_results["labels"].append(res_i["no_model_batch"])
            
        merge_results = dict()

        merge_results["input_ids"] = torch.concatenate(collected_results["input_ids"], dim=0)
        merge_results["attention_mask"] = torch.concatenate(collected_results["attention_mask"], dim=0)
        merge_results["position_ids"] = torch.concatenate(collected_results["position_ids"], dim=0)
        merge_results["labels"] = torch.concatenate(collected_results["labels"], dim=0)
        merge_results["loss_mask"] = merge_results["labels"] != -100

        return merge_results, diff, mean_rouge_score 
        
    


    
