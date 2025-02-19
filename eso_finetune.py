import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import deepspeed

import random
import json
from tqdm import tqdm
import math
import datetime

from transformers import (
    AutoModelForCausalLM,
    OPTForCausalLM,
    AutoTokenizer,
    AutoConfig,
    # mpu,
    GenerationConfig)

from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR

from arguments import get_args

from data_utils.lm_datasets import LMTrainDataset
from utils import get_optimizer_params, print_args, initialize
from utils import print_rank, get_rank
from utils import save_rank
from utils import all_gather
# from utils import load_parallel, save_parallel
from utils import get_tokenizer, get_model

from sampler import NesSampleGenerator

from accelerate import init_empty_weights

from rouge_metric import compute_metrics

torch.set_num_threads(4)


def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)


def get_optimizer(args, model):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, DDP):
        model = model.module

    
    param_groups = get_optimizer_params(args, model)

    # Use AdamW.
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    print_rank(f'Optimizer = {optimizer.__class__.__name__}')
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.total_iters is None:
        args.total_iters = args.train_iters_per_epoch * args.epochs
    if args.lr_decay_style == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters)
    elif args.lr_decay_style == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.total_iters,
            eta_min=args.lr_min)
    elif args.lr_decay_style == "noam":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters,
            num_training_steps=args.total_iters,
            power=0.5)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler


def setup_model_and_optimizer(args, ds_config, device, set_optim=True):
    # get the model
    model = get_model(args, device)
    # get the optimizer and lr_scheduler
    if set_optim:
        optimizer = get_optimizer(args, model)
        lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    else:
        optimizer, lr_scheduler = None, None
        
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        # mpu=mpu if args.model_parallel else None,
        config_params=ds_config
    )
    
    # get the memory usage
    print_rank("Model mem\n", torch.cuda.memory_summary())
    return model, optimizer, lr_scheduler


def prepare_dataset(args, tokenizer):
    data = {}
    rng_sample = random.Random(args.seed)
    if args.do_train:
        data["train"] = LMTrainDataset(args, tokenizer, args.data_dir, "train", args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["train"]))
        data["dev"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
    elif args.do_eval:
        data["test"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
    else:
        raise ValueError("Do train and do eval must set one")
        
    # pre-trained dataset
    if args.do_train and args.lm_data_dir is not None:
        data["pt_train"] = LMTrainDataset(args, tokenizer, args.lm_data_dir, "train", args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["pt_train"]))
    return data


def finetune(args, tokenizer: AutoTokenizer, model: deepspeed.DeepSpeedEngine, optimizer: AdamW, lr_scheduler, dataset, device, teacher_model=None):
    print_rank("Start Fine-tuning")

    # print_inspect(model, '*')
    if args.model_parallel:
        raise NotImplementedError
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        loss_func = nn.CrossEntropyLoss()

    sft_sampler = DistributedSampler(dataset["train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
    train_dataloader = DataLoader(
        dataset['train'], sampler=sft_sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset["train"].collate)

    online_generator = NesSampleGenerator(args, tokenizer)

    step, global_step = 1, 1
    total_loss, total_lm_loss, total_nes_loss, total_mean_reward, total_time = 0.0, 0.0, 0.0, 0.0, 0.0
    
    # evaluate(args, tokenizer, model, dataset["dev"], "dev", 0, device)
    
    for epoch in range(args.epochs):
        sft_sampler.set_epoch(epoch)

        model.train()
        for it, (model_batch, no_model_batch, gen_data) in enumerate(train_dataloader):
            dataset["train"].move_to_device(model_batch, no_model_batch, gen_data, device)
            
            torch.cuda.synchronize()
            st_time = time.time()

            outputs = model(**model_batch, use_cache=False)

            logits = outputs.logits
            if args.model_parallel:
                raise NotImplementedError
            else:
                lm_loss = loss_func(logits.float().view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))

            # if epoch == args.epochs - 1:
            if epoch < args.epochs - 1:
                # data generation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
                assert args.online_gen
                
                # print(gen_data.keys())
                if args.nes_sample == "temp":
                    concanated_batch, diff, mean_reward = online_generator.run_sample_temp(model, gen_data, model_batch)
                else:
                    raise NotImplementedError

                concanated_no_model_batch = dict()
                concanated_no_model_batch["labels"] = concanated_batch.pop("labels")
                concanated_no_model_batch["loss_mask"] = concanated_batch.pop("loss_mask")

                if "gpt" not in args.model_type:
                    concanated_batch.pop('position_ids')
                    
                model.train()
                
                concanated_outputs = model(**concanated_batch, use_cache=False)

                concanated_logits = concanated_outputs.logits # (bs, sequence_length, num_vocs)

                average_log_prob = True if args.length_norm else False

                all_logps = get_batch_logps(
                                concanated_logits,
                                concanated_no_model_batch["labels"],
                                average_log_prob=average_log_prob,
                                is_encoder_decoder=False,
                                label_pad_token_id=-100)

                all_logps = all_logps.reshape(concanated_batch["input_ids"].shape[0], -1)

                all_logps_split = torch.split(all_logps, model_batch["input_ids"].size(0), dim=0)

                reconcanated_logps = torch.concatenate(all_logps_split, dim=1)

                
                if args.reward_thres is not None:
                    reward_mask = mean_reward > args.reward_thres
                    nes_loss = torch.sum(torch.sum(diff * reconcanated_logps, dim=1)*reward_mask) / (torch.sum(reward_mask) + 1e-12)
                else:
                    nes_loss = torch.mean(torch.sum(diff * reconcanated_logps, dim=1))

            else:
                # print("epoch:", epoch)
                nes_loss = torch.tensor(0.0).to(device)
                mean_reward = torch.tensor(0.0).to(device)

            loss = lm_loss + args.nes_coef*nes_loss

            model.backward(loss)
            model.step()
             
            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            global_loss = loss.item() / dp_world_size
            # print("global_loss:", global_loss)

            dist.all_reduce(lm_loss, dist.ReduceOp.SUM, group=dp_group)
            global_lm_loss = lm_loss.item() / dp_world_size
            # print("global_lm_loss:", global_lm_loss)

            dist.all_reduce(nes_loss, dist.ReduceOp.SUM, group=dp_group)
            global_nes_loss = nes_loss.item() / dp_world_size

            dist.all_reduce(mean_reward, dist.ReduceOp.SUM, group=dp_group)
            global_mean_reward = mean_reward.mean().item() / dp_world_size
    
            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time

        
            total_loss += global_loss

            total_lm_loss += global_lm_loss

            total_nes_loss += global_nes_loss

            total_mean_reward += global_mean_reward

            total_time += elapsed_time

            # Logging
            def get_log(log_loss, lm_loss, nes_loss, mean_reward, log_time):
                return "train | epoch {:3d} | Iter: {:6d}/{:6d} | global iter: {:6d}/{:6d} | loss: {:.4f} | lm_loss: {:.4f}| nes_loss: {:.4f}| mean_reward: {:.4f} | lr: {:.4e} | scale: {:10.4f} | micro time: {:.3f} | step time: {:.3f}".format(
                    epoch,
                    step,
                    args.total_iters * args.gradient_accumulation_steps,
                    global_step,
                    args.total_iters,
                    log_loss,
                    lm_loss,
                    nes_loss,
                    mean_reward,
                    lr_scheduler.get_last_lr()[0],
                    optimizer.cur_scale if hasattr(optimizer, "cur_scale") else 0,
                    elapsed_time,
                    log_time,
                )

            if args.mid_log_num > 0:
                mid_log_step = args.gradient_accumulation_steps // args.mid_log_num
                mid_log_step = 1 if mid_log_step == 0 else mid_log_step
                if step % mid_log_step == 0:
                    print_rank(get_log(global_loss, global_lm_loss, global_nes_loss, global_mean_reward, 0))

            if global_step % args.log_interval == 0 and step % args.gradient_accumulation_steps == 0:
                log_str = get_log(
                    total_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_lm_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_nes_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_mean_reward / (args.log_interval * args.gradient_accumulation_steps),
                    total_time / (args.log_interval))
                print_rank("*" * 100)
                print_rank(log_str)
                print_rank(args.save)
                print_rank("*" * 100)
                save_rank(log_str, os.path.join(args.save, "log.txt"))
                total_loss, total_lm_loss, total_nes_loss, total_mean_reward, total_time = 0.0, 0.0, 0.0, 0.0, 0.0
            
            # Checkpointing
            if args.save and args.save_interval and global_step % args.save_interval == 0 and step % args.gradient_accumulation_steps == 0:
                save_dir_path = os.path.join(args.save, str(global_step))
                if args.model_parallel:
                    raise NotImplementedError
                else:
                    if dist.get_rank() == 0:
                        os.makedirs(save_dir_path, exist_ok=True)
                        print_rank(f"Model save to {save_dir_path}")
                        tokenizer.save_pretrained(save_dir_path)
                        model.module.save_pretrained(save_dir_path, safe_serialization=False)
                dist.barrier()

            # Evaluation
            if args.eval_interval and global_step % args.eval_interval == 0 and step % args.gradient_accumulation_steps == 0:
                evaluate(args, tokenizer, model, dataset["dev"], "dev", epoch, device)
                    
                model.train()
                
            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1
            
            if global_step > args.total_iters:
                break
            
    return model


def evaluate(args, tokenizer, model, dataset: LMTrainDataset, split, epoch, device):
    
    collate_fn = dataset.collate

    if args.model_parallel:
        raise NotImplementedError
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        loss_func = nn.CrossEntropyLoss()

    print_rank("dp size", dp_world_size)

    top_k_list = [0, 5, 10]
    # top_k_list = [0]

    for top_k in top_k_list:

        generation_config = GenerationConfig(
            do_sample=args.do_sample,
            top_p=args.top_p,
            top_k=top_k,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            # max_length=args.max_length,
            min_length=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False
        )

        sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
        dataloader = DataLoader(
            dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

        model.eval()
        all_loss = 0.0
        step = 0
        
        all_response_ids = []
        
        with torch.no_grad():
            for it, (model_batch, no_model_batch, gen_data) in enumerate(tqdm(dataloader, desc="Evaluating", disable=(dist.get_rank() != 0))):
                print_rank(f"{it}/{len(dataloader)}")
                dataset.move_to_device(model_batch, no_model_batch, gen_data, device)
                logits = model(**model_batch).logits
                if args.model_parallel:
                    raise NotImplementedError
                else:
                    loss = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
                    
                gen_data.pop("target_ids")

                max_new_tokens = args.max_new_tokens
                
                if args.eval_gen:            
                    gen_out = model.generate(
                        **gen_data,
                        generation_config=generation_config,
                        max_new_tokens=max_new_tokens)
                    
                    full_ids = gen_out.sequences
                    
                    full_ids = F.pad(
                        full_ids,
                        (0, args.max_length + max_new_tokens - full_ids.shape[1]),
                        value=tokenizer.pad_token_id,
                    )
                    
                    response_ids = full_ids[:, gen_data["input_ids"].size(1):]
                    all_response_ids.append(response_ids)
                        
                dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
                loss = loss / dp_world_size
                all_loss += loss.item()
                step += 1
        
        if args.eval_gen:
            all_response_ids = torch.cat(all_response_ids, dim=0)
            all_response_ids = all_gather(all_response_ids, dim=1, world_size=dp_world_size, group=dp_group, op="stack")
            all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))
            
            responses = tokenizer.batch_decode(all_response_ids, skip_special_tokens=True)
        
        if get_rank() == 0:
            if args.eval_gen:
                references = dataset.answers
                responses = responses[:len(references)]
                
                res = compute_metrics(responses, references)
                # assert 0
            
                eval_dir = os.path.join(args.save, "eval", str(epoch))
                print_rank(eval_dir)
                os.makedirs(eval_dir, exist_ok=True)
                with open(os.path.join(eval_dir, "answers_topk(%s).jsonl"%top_k), "w") as f:
                    for resp in responses:
                        f.write(json.dumps({"response": resp}) + "\n")
            else:
                res = {}
        
            avg_loss = all_loss / step
            
            log_str = f"topk:{top_k} | {split} | avg_loss: {avg_loss} | {res}"
            print_rank(log_str)
            save_rank(log_str, os.path.join(args.save, "log.txt"))
        
    return all_loss / step


def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    initialize(args)
    
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000
    
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0
    try:
        args.fp32 = not ds_config["fp16"]["enabled"]
    except:
        args.fp32 = False

    args.deepspeed_config = None
    
    # get the tokenizer
    tokenizer = get_tokenizer(args)
    dataset = prepare_dataset(
        args,
        tokenizer,
    )
    
    # dp_world_size = mpu.get_data_parallel_world_size() if args.model_parallel else dist.get_world_size()
    dp_world_size = dist.get_world_size()
    
    if args.do_train:
        args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dp_world_size * args.gradient_accumulation_steps))
        print_rank("Train iters per epoch", args.train_iters_per_epoch)
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.epochs
        if args.epochs is None:
            args.epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)
        print_rank("total_iters", args.total_iters)
        
        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch
        
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch
    
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, ds_config, device, set_optim=args.do_train)
    
    if args.do_train:
        model = finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, device)
   
    if args.do_eval:
        evaluate(args, tokenizer, model, dataset["test"], "test", 0, device)
        
if __name__ == "__main__":
    main()
