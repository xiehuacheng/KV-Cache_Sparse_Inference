import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaAttention, LlamaDecoderLayer, \
                                                     apply_rotary_pos_emb, repeat_kv, rotate_half
from lm_eval.models.huggingface import HFLM

from utils.quantizer import UniformAffineQuantizer

from transformers.cache_utils import Cache, SinkCache
from utils.cacheutils import EditableDynamicCache, InitializedSinkCache

from optimize_utils.time_counter import TimeCounter

from tqdm import tqdm

def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization! 
    pass


class IntactKVLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, model: LlamaForCausalLM, tokenizer,
                 intactkv, intactkv_ids, intactkv_logits):
        super(LlamaForCausalLM, self).__init__(model.config)
        self.model = model.model
        self.lm_head = model.lm_head
        self.config = model.config
        if hasattr(model, "seqlen"):
            self.seqlen = model.seqlen

        # intactKV
        self.intactkv = intactkv
        self.intactkv_ids = intactkv_ids
        self.intactkv_logits = intactkv_logits
        if intactkv is not None:
            self.intactkv_len = self.intactkv_ids.shape[-1]
            dtype = next(self.model.parameters()).dtype
            # move intactKV to GPU
            self.intactkv = []
            for i, layer_kv in enumerate(intactkv):
                gpu_layer_kv = []
                device = next(self.model.layers[i].parameters()).device
                for cache in layer_kv:
                    gpu_layer_kv.append(cache.to(device).to(dtype))
                self.intactkv.append(tuple(gpu_layer_kv))
            self.intactkv = tuple(self.intactkv)
            self.intactkv_logits = self.intactkv_logits.to(device)
        else:
            self.intactkv_len = None
            device = next(self.model.parameters()).device
            self.bos_ids = tokenizer(tokenizer.bos_token, return_tensors='pt', add_special_tokens=False).input_ids.to(device)

        self.enable_streaming_llm_kv_cache = False
        self.enable_h2o_kv_cache = False
        self.auto_regressive_mode = False
        
        self.kv_cache = None
            
        # streaming llm kv cache
        self.start_size = 0
        self.recent_size = 0
    
    def enable_auto_regressive(self):
        self.auto_regressive_mode = True
    
    def enable_streaming_llm(self, start_size, recent_size):
        # 取 startsize 为 startsize 和 intactkvlen 的较大值
        if self.intactkv_len is not None:
            start_size = max(start_size, self.intactkv_len)
        self.enable_streaming_llm_kv_cache = True
        self.start_size = start_size
        self.recent_size = recent_size

    def enable_h2o(self):
        self.enable_h2o_kv_cache = True
        
    def init_intactkv_and_cache(self, input_ids):
        
        batch_size = input_ids.shape[0]
        
        # 初始化 intactKV
        activate_intactkv = self.intactkv is not None
        
        if activate_intactkv:
            # 在 batch 维度上复制 batch_size 次
            self.intactkv_ids = self.intactkv_ids.repeat(batch_size, 1)
            if (input_ids[:, :self.intactkv_len] == self.intactkv_ids.to(input_ids.device)).all():
                input_ids = input_ids[:, self.intactkv_len:]
            self.kv_cache = self.intactkv
        else:
            # 在 batch 维度上复制 batch_size 次
            self.bos_ids = self.bos_ids.repeat(batch_size, 1)
            # prepend <bos>
            if not (input_ids[:, :1] == self.bos_ids).all():
                input_ids = torch.cat([self.bos_ids, input_ids], dim=1).contiguous()
        
        init_past_key_values = None
        if self.enable_h2o_kv_cache:
            init_past_key_values = EditableDynamicCache.from_legacy_cache(self.kv_cache)
        else:
            if self.enable_streaming_llm_kv_cache:
                init_past_key_values = InitializedSinkCache(self.start_size + self.recent_size, self.start_size)
                init_past_key_values.from_legacy_cache(self.kv_cache)
        
        self.kv_cache = init_past_key_values
        
        return input_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # TODO. support bs=1 only
        # assert input_ids.shape[0] == 1
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        input_ids = input_ids.to(self.device)
        
        activate_intactkv = (past_key_values is None and self.intactkv is not None)

        input_len = input_ids.shape[1]
        
        if past_key_values is None:
            input_ids = self.init_intactkv_and_cache(input_ids)
        
        if self.auto_regressive_mode:
        
            seq_len = input_ids.shape[1]
                    
            logits_list = []

            pbar = tqdm(range(0, seq_len))
            
            for idx in pbar:
                
                step_input_ids = input_ids[:, idx : idx + 1]
                            
                # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
                # 模型前向推理代码实现位置，在这里添加 kv_cache 以及 streamingllm 的替换规则
                outputs = self.model(
                    input_ids=step_input_ids,
                    attention_mask=None,
                    position_ids=None,
                    past_key_values=self.kv_cache,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=return_dict,
                )

                self.kv_cache = outputs.past_key_values

                hidden_states = outputs[0]
                logits = self.lm_head(hidden_states)
                logits_list.append(logits)

            # print(len(logits_list))
            logits = torch.cat(logits_list, dim=1)

            if self.enable_h2o_kv_cache:
                for name, m in self.model.named_modules():
                    if isinstance(m, LlamaDecoderLayer):
                        m.self_attn.kv_cache._clean_scores()
                        # if m.self_attn.layer_idx == 0:
                            # for i, timer in enumerate(m.self_attn.timers):
                            #     print(f"timer {i}:")
                            #     timer.print_statistics()
                            #     timer.reset()
                            # for i, timer in enumerate(m.self_attn.kv_cache.timers):
                            #     print(f"timer {i}:")
                            #     timer.print_statistics()
                            #     timer.reset()

        else:
            if isinstance(self.kv_cache, SinkCache):
                # 对输入的 token 进行截断
                num_sink_tokens = self.kv_cache.num_sink_tokens
                window_length = self.kv_cache.window_length
                recent_size = window_length - num_sink_tokens
                if input_len > window_length:
                    # 将 input_ids 进行截断，保留最前面的 num_sink_tokens 个 token 以及 最近 recent_size 个 token
                    sink = input_ids[:, :num_sink_tokens]
                    recent = input_ids[:, -recent_size:]
                    input_ids = torch.cat([sink, recent], dim=1)
            
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=self.kv_cache,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=return_dict,
            )

            self.kv_cache = outputs.past_key_values
            
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)

            if self.enable_h2o_kv_cache:
                for name, m in self.model.named_modules():
                    if isinstance(m, LlamaDecoderLayer):
                        m.self_attn.kv_cache._clean_scores()
        
        # print(logits.shape)
        if activate_intactkv:
            logits = torch.cat([self.intactkv_logits.to(logits), logits], dim=1)
        logits = logits.float()
        logits = logits[:, -input_len:, :]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def inject_intactkv_inference_model(args, model, tokenizer, intactkv, intactkv_ids, intactkv_logits):
    # locate causal LM
    if isinstance(model, HFLM):
        if args.quant_method == "gptq":
            causal_lm = model.model.model
        else:
            causal_lm = model.model
    else:
        causal_lm = model

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    # intactKV model
    intactkv_params = (causal_lm, tokenizer, intactkv, intactkv_ids, intactkv_logits)
    if isinstance(causal_lm, LlamaForCausalLM):
        causal_lm = IntactKVLlamaForCausalLM(*intactkv_params)
        if args.enable_streaming_pos_shift:
            causal_lm.enable_streaming_llm(args.start_size, args.recent_size)
        if args.enable_h2o:
            causal_lm.enable_h2o()
    else:
        raise NotImplementedError
    
    # inject model
    if isinstance(model, HFLM):
        if args.quant_method == "gptq":
            model.model.model = causal_lm
        else:
            model._model = causal_lm
    else:
        model = causal_lm
    
    return model


class LlamaAttentionKVQuantized(LlamaAttention):
    def __init__(self, args, m: LlamaAttention):
        super().__init__(m.config)
        self.layer_idx = m.layer_idx
        
        self.q_proj = m.q_proj
        self.k_proj = m.k_proj
        self.v_proj = m.v_proj
        self.o_proj = m.o_proj
        self.rotary_emb = m.rotary_emb

        # KV cache quantizer
        self.quant_params = {
            "n_bits": args.kv_bits,
            "symmetric": False,
        }
        self.k_quantizer = UniformAffineQuantizer(**self.quant_params)
        self.v_quantizer = UniformAffineQuantizer(**self.quant_params)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # quantize KV cache
        key_states = self.k_quantizer(key_states)
        value_states = self.v_quantizer(value_states)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def inject_quantized_kv_model(args, model: HFLM):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    # TODO. support pseudo-quantized model only
    if args.quant_method == "gptq":
        raise NotImplementedError
    
    causal_lm = model.model

    # intactKV model
    if isinstance(causal_lm, LlamaForCausalLM):
        for name, m in causal_lm.named_modules():
            if isinstance(m, LlamaDecoderLayer):
                m.self_attn = LlamaAttentionKVQuantized(args, m.self_attn)
    else:
        raise NotImplementedError
    
    model._model = causal_lm
    
    return model

# =======================================================================================================================
# Streaming
# =======================================================================================================================

# 单独应用旋转位置嵌入
def apply_rotary_pos_emb_single(x, cos, sin, position_ids, unsqueeze_dim=1):
    # cos和sin的前两维始终是1，因此可以用`squeeze`去掉这两维
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    # 通过旋转位置嵌入修改输入的x
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

class LlamaAttentionWithStreaming(LlamaAttention):
    def __init__(self, args, m: LlamaAttention):
        super().__init__(m.config)
        self.layer_idx = m.layer_idx
        
        self.q_proj = m.q_proj
        self.k_proj = m.k_proj
        self.v_proj = m.v_proj
        self.o_proj = m.o_proj
        self.rotary_emb = m.rotary_emb
        
        if args.kv_bits < 16:
            # KV cache quantizer
            self.quant_params = {
                "n_bits": args.kv_bits,
                "symmetric": False,
            }
            self.k_quantizer = UniformAffineQuantizer(**self.quant_params)
            self.v_quantizer = UniformAffineQuantizer(**self.quant_params)
        else:
            self.k_quantizer = None
            self.v_quantizer = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if self.k_quantizer is not None:
            # quantize KV cache
            key_states = self.k_quantizer(key_states)
            value_states = self.v_quantizer(value_states)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

def inject_streaming_model(args, model: HFLM):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    causal_lm = model.model

    # intactKV model
    if isinstance(causal_lm, LlamaForCausalLM):
        for name, m in causal_lm.named_modules():
            if isinstance(m, LlamaDecoderLayer):
                m.self_attn = LlamaAttentionWithStreaming(args, m.self_attn)
    else:
        raise NotImplementedError
    
    model._model = causal_lm
    
    return model

# =======================================================================================================================
# H2O
# =======================================================================================================================

def _make_causal_mask(
    bsz: int, tgt_len: int, past_key_values_length: int, dtype: torch.dtype, device: torch.device):
    """
    Make causal mask used for bi-directional self-attention.
    """
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

class H2OKVCache_LayerWise:
    def __init__(
        self,
        num_attention_heads,
        num_key_value_heads,
        device,
        batch_size,
        k=4,
        hh_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        print(f"H2OKVCache-LayerWise: {hh_size}, {recent_size}")
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.group_size = num_attention_heads // num_key_value_heads # 多少个 Query 头共用一组 KVCache
        self.k = k
        self.min_k = hh_size + 1 - k
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.cache_size = hh_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None
        
        # self.timers = []
        
        # for i in range(9):
        #     tm = TimeCounter()
        #     self.timers.append(tm)
        
        self.keep_recent = torch.arange(self.hh_size + 1, self.cache_size + 1, device=device).repeat(batch_size, num_key_value_heads, 1)
        # self.hh_score = torch.zeros(size=(batch_size, num_key_value_heads, 1), device=device)

        self.decay_rate = 1

    def __call__(self, past_key_values, attn_score_cache):
        
        with torch.no_grad():
        
            # self.timers[3].start()
            self._update_hh_score(attn_score_cache)
            # self.timers[3].stop()

            # self.timers[4].start()
            if past_key_values is None:
                # self.timers[4].stop()
                return None
            seq_len = past_key_values[0].size(self.k_seq_dim)
            if seq_len <= self.cache_size:
                # self.timers[4].stop()
                return past_key_values
            # self.timers[4].stop()

            # self.timers[5].start()
            # hh-selection
            bsz, num_heads, _, head_dim = past_key_values[0].shape

            select_hh_scores = self.hh_score[:, :, :seq_len - self.recent_size]
            # self.timers[5].stop()
            
            # self.timers[6].start()
            keep_topk = torch.topk(select_hh_scores, self.k, dim=-1, sorted=False).indices.sort().values
            # _, keep_topk = torch.topk(select_hh_scores, self.k, dim=-1)
            # keep_topk = keep_topk.sort().values
            # self.timers[6].stop()

            # self.timers[7].start()
            # keep_recent = torch.arange(seq_len - self.recent_size, seq_len, device=keep_topk.device).repeat(keep_topk.shape[0], keep_topk.shape[1], 1)
            # keep_idx = torch.cat([keep_topk, self.keep_recent], dim=-1)        
            keep_idx_3d = torch.cat([keep_topk, self.keep_recent], dim=-1)
            keep_idx_4d = keep_idx_3d.unsqueeze(-1).repeat(1, 1, 1, head_dim)
            
            # mask = torch.zeros(self.hh_score.shape, dtype=torch.bool).to(past_key_values[0].device)
            # mask = mask.scatter(-1, keep_idx, 1)
            # self.timers[7].stop()

            # self.timers[8].start()
            k_hh_recent = past_key_values[0].gather(2, keep_idx_4d)
            v_hh_recent = past_key_values[1].gather(2, keep_idx_4d)

            self.hh_score = self.hh_score.gather(2, keep_idx_3d)
            # k_hh_recent = past_key_values[0].squeeze()[mask].v
            # iew(bsz, num_heads, -1, head_dim)
            # v_hh_recent = past_key_values[1].squeeze()[mask].view(bsz, num_heads, -1, head_dim)

            # self.hh_score= self.hh_score[mask].view(bsz, num_heads, self.k + self.recent_size)
            # self.timers[8].stop()

            return (k_hh_recent, v_hh_recent)

    def _update_hh_score(self, attn_score_cache):
        with torch.no_grad():
        
            # self.timers[0].start()

            num_new_tokens = attn_score_cache.shape[2]

            attn_score_cache = attn_score_cache.view(attn_score_cache.shape[0], self.group_size, -1, attn_score_cache.shape[2], attn_score_cache.shape[3])

            # bsz, group_size, num_kv_heads, num_new_tokens, seq_len

            # self.timers[0].stop()
            # self.timers[1].start()
            
            # self.hh_score = _sum_and_merge(attn_score_cache, self.hh_score, num_new_tokens)
            
            # self.timers[1].stop()

            if self.hh_score is None:
                self.hh_score = attn_score_cache.sum(dim=(1, 3))
                # self.hh_score = attn_score_cache.sum(1).sum(2)
                # self.timers[1].stop()
                
            else:
                attn_score_cache = attn_score_cache.sum(dim=(1, 3))  # 直接对第1、2维度求和
                # attn_score_cache = attn_score_cache.sum(1).sum(2)
                # self.timers[1].stop()
                
                # self.timers[2].start()
                # attn_score_cache[:, :, :-num_new_tokens].add_(self.hh_score, alpha=self.decay_rate)
                attn_score_cache[:, :, :-num_new_tokens].add_(self.hh_score)
                # attn_score_cache[:, :, :-num_new_tokens] += self.hh_score
                self.hh_score = attn_score_cache
                # self.timers[2].stop()

    def _clean_scores(self):
        self.hh_score = None
    
    def evict_for_space(self, past_key_values, num_coming):
        
        if past_key_values is None:
            return None
        seq_len = past_key_values[-1].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values

        # hh-selection
        bsz, num_heads, _, head_dim = past_key_values[0].shape

        select_hh_scores = self.hh_score[:, :, :seq_len - self.recent_size + num_coming]
        keep_topk = torch.topk(select_hh_scores, self.hh_size, dim=-1).indices.sort().values
        
        keep_recent = torch.arange(seq_len - self.recent_size + num_coming, seq_len, device=keep_topk.device).repeat(keep_topk.shape[0], keep_topk.shape[1], 1)

        keep_idx_3d = torch.cat([keep_topk, keep_recent], dim=-1)
        keep_idx_4d = keep_idx_3d.unsqueeze(-1).repeat(1, 1, 1, head_dim)
        
        k_hh_recent = past_key_values[0].gather(2, keep_idx_4d)
        v_hh_recent = past_key_values[1].gather(2, keep_idx_4d)

        self.hh_score = self.hh_score.gather(2, keep_idx_3d)

        return (k_hh_recent, v_hh_recent)


class H2OLlamaAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, args, m: LlamaAttention):
        super().__init__(m.config)
        self.layer_idx = m.layer_idx
        
        self.q_proj = m.q_proj
        self.k_proj = m.k_proj
        self.v_proj = m.v_proj
        self.o_proj = m.o_proj
        self.rotary_emb = m.rotary_emb

        self.kv_cache = H2OKVCache_LayerWise(
            num_attention_heads=m.config.num_attention_heads,
            num_key_value_heads=m.config.num_key_value_heads,
            device="cuda",
            batch_size=args.batch_size,
            k = args.k,
            hh_size=args.heavy_hitter_size,
            recent_size=args.heavy_hitter_recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
        )
        
        self.timers = []
        
        for i in range(7):
            tm = TimeCounter()
            self.timers.append(tm)
        
        if args.kv_bits < 16:
            # KV cache quantizer
            self.quant_params = {
                "n_bits": args.kv_bits,
                "symmetric": False,
            }
            self.k_quantizer = UniformAffineQuantizer(**self.quant_params)
            self.v_quantizer = UniformAffineQuantizer(**self.quant_params)
        else:
            self.k_quantizer = None
            self.v_quantizer = None            
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        # self.timers[0].start()
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        
        temp_len = 0

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            temp_len = past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            kv_seq_len += temp_len
        
        # self.timers[0].stop()
        
        # self.timers[1].start()
        
        # remake causal mask
        
        if q_len != 1:
            attention_mask = _make_causal_mask(
                bsz=bsz,
                tgt_len=q_len,
                past_key_values_length=temp_len,
                dtype=query_states.dtype,
                device=query_states.device,
            )
        
        # self.timers[1].stop()
        
        # self.timers[2].start()

        # position_length = kv_seq_len
        # if not position_ids.nelement() > 1:
        #     if position_length < position_ids.item()+1:
        #         position_length = position_ids.item()+1

        # cos, sin = self.rotary_emb(value_states, seq_len=position_length)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        ### Shift Pos: query pos is min(cache_size, idx)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        # key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)
        
        if self.k_quantizer is not None:
            # quantize KV cache
            key_states = self.k_quantizer(key_states)
            value_states = self.v_quantizer(value_states)
            
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        # self.timers[2].stop()
        
        # temp_past_key_value = past_key_value.to_legacy_cache()

        
        # self.timers[3].start()

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        
        # self.timers[3].stop()
        
        # self.timers[4].start()
        # past_key_value.replace(self.kv_cache(temp_past_key_value[self.layer_idx], attn_weights.detach().clone()), self.layer_idx)

        past_key_value.key_cache[self.layer_idx], past_key_value.value_cache[self.layer_idx] = self.kv_cache((past_key_value.key_cache[self.layer_idx], past_key_value.value_cache[self.layer_idx]), attn_weights.detach().clone())
        
        # self.timers[4].stop()
        # self.timers[5].start()

        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        
        # self.timers[5].stop()

        return attn_output, attn_weights, past_key_value
    
def inject_h2o_model(args, model: HFLM):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    causal_lm = model.model

    # intactKV model
    if isinstance(causal_lm, LlamaForCausalLM):
        for name, m in causal_lm.named_modules():
            if isinstance(m, LlamaDecoderLayer):
                m.self_attn = H2OLlamaAttention(args, m.self_attn)
    else:
        raise NotImplementedError
    
    model._model = causal_lm
    
    return model

## H2O KV Cache dropping with Position rolling
class H2OLlamaAttentionWithStreaming(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, args, m: LlamaAttention):
        super().__init__(m.config)
        self.layer_idx = m.layer_idx
        
        self.q_proj = m.q_proj
        self.k_proj = m.k_proj
        self.v_proj = m.v_proj
        self.o_proj = m.o_proj
        self.rotary_emb = m.rotary_emb

        self.kv_cache = H2OKVCache_LayerWise(
            num_attention_heads=m.config.num_attention_heads,
            num_key_value_heads=m.config.num_key_value_heads,
            device="cuda",
            batch_size=args.batch_size,
            k = args.k,
            hh_size=args.heavy_hitter_size,
            recent_size=args.heavy_hitter_recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
        )
        
        if args.kv_bits < 16:
            # KV cache quantizer
            self.quant_params = {
                "n_bits": args.kv_bits,
                "symmetric": False,
            }
            self.k_quantizer = UniformAffineQuantizer(**self.quant_params)
            self.v_quantizer = UniformAffineQuantizer(**self.quant_params)
        else:
            self.k_quantizer = None
            self.v_quantizer = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        
        temp_len = 0

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            temp_len = past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            kv_seq_len += temp_len
        
        # remake causal mask
        attention_mask = _make_causal_mask(
            bsz=bsz,
            tgt_len=q_len,
            past_key_values_length=temp_len,
            dtype=query_states.dtype,
            device=query_states.device,
        )

        if not position_ids.nelement() > 1:
            position_ids[0][0] = kv_seq_len - 1

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        ### Shift Pos: query pos is min(cache_size, idx)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        
        if self.k_quantizer is not None:
            # quantize KV cache
            key_states = self.k_quantizer(key_states)
            value_states = self.v_quantizer(value_states)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # print(f"{self.layer_idx}:", past_key_value[0][0].shape)
        temp_past_key_value = past_key_value.to_legacy_cache()
        # input()

        ### Shift Pos: key pos is the pos in cache (Rolling KV Cache and using relative pos emb)
        key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)
        ###

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        # print("attn_weights:", attn_weights.shape)
        past_key_value.replace(self.kv_cache(temp_past_key_value[self.layer_idx], attn_weights.detach().clone()), self.layer_idx)
        # print(f"{self.layer_idx}:", past_key_value[0][0].shape)

        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

def inject_h2o_streaming_model(args, model: HFLM):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    causal_lm = model.model

    # intactKV model
    if isinstance(causal_lm, LlamaForCausalLM):
        for name, m in causal_lm.named_modules():
            if isinstance(m, LlamaDecoderLayer):
                m.self_attn = H2OLlamaAttentionWithStreaming(args, m.self_attn)
    else:
        raise NotImplementedError
    
    model._model = causal_lm
    
    return model
