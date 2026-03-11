import warnings

from transformers.models.llama.modeling_llama import LlamaAttention
from utils.cacheutils import EditableDynamicCache, InitializedSinkCache

warnings.filterwarnings("ignore")

import torch
import os

from streaming_llm.utils import load, download_url, load_jsonl

@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, max_gen_len):
    outputs = model.model(
        input_ids=input_ids,
        past_key_values=model.kv_cache,
        use_cache=True,
    )
    
    model.kv_cache = outputs.past_key_values
    hidden_states = outputs[0]
    logits = model.lm_head(hidden_states)
    
    pred_token_idx = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0
    for _ in range(max_gen_len - 1):
        outputs = model.model(
            input_ids=pred_token_idx,
            past_key_values=model.kv_cache,
            use_cache=True,
        )
        model.kv_cache = outputs.past_key_values
        hidden_states = outputs[0]
        logits = model.lm_head(hidden_states)
        pred_token_idx = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        generated_text = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )

        now = len(generated_text) - 1
        if now > pos:
            print(" ".join(generated_text[pos:now]), end=" ", flush=True)
            pos = now

        if pred_token_idx == tokenizer.eos_token_id:
            break
    print(" ".join(generated_text[pos:]), flush=True)

@torch.no_grad()
def streaming_inference(model, prompts, max_gen_len=1000):
    for idx, prompt in enumerate(prompts):
        
        if idx == 5:
            break
        
        prompt = "USER: " + prompt + "\n\nASSISTANT: "
        print("\n" + prompt, end="")
        input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to("cuda")
        seq_len = input_ids.shape[1]
        if model.model.kv_cache is not None:
            space_needed = seq_len
            if isinstance(model.model.kv_cache, EditableDynamicCache):
                for name, m in model.model.named_modules():
                    if isinstance(m, LlamaAttention):
                        layer_idx = int(name.split(".")[2])
                        model.model.kv_cache.key_cache[layer_idx], model.model.kv_cache.value_cache[layer_idx] = m.kv_cache.evict_for_space((model.model.kv_cache.key_cache[layer_idx], model.model.kv_cache.value_cache[layer_idx]), space_needed)
        else:
            input_ids = model.model.init_intactkv_and_cache(input_ids)

        greedy_generate(
            model.model, model.tokenizer, input_ids, max_gen_len=max_gen_len
        )

def streaming_eval(model):
    data_root = "./datasets"
    test_filepath = os.path.join(data_root, "mt_bench.jsonl")
    print(f"Loading data from {test_filepath} ...")

    if not os.path.exists(test_filepath):
        download_url(
            "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
            data_root,
        )
        os.rename(os.path.join(data_root, "question.jsonl"), test_filepath)

    list_data = load_jsonl(test_filepath)
    prompts = []
    for idx, sample in enumerate(list_data):
        prompts += sample["turns"]

        if idx == 1:
            break

        streaming_inference(
            model,
            prompts,
        )
