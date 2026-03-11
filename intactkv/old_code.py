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
        k = 3,
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
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.cache_size = hh_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None

    def __call__(self, past_key_values, attn_score_cache):

        self._update_hh_score(attn_score_cache)

        if past_key_values is None:
            return None
        seq_len = past_key_values[0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values

        # hh-selection
        bsz, num_heads, _, head_dim = past_key_values[0].shape

        select_hh_scores = self.hh_score[:, :seq_len - self.recent_size]
        _, keep_topk = torch.topk(select_hh_scores, self.k, dim=-1)
        keep_topk = keep_topk.sort().values

        keep_recent = torch.arange(seq_len - self.recent_size, seq_len, device=keep_topk.device).repeat(keep_topk.shape[0], 1)
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)

        mask = torch.zeros(self.hh_score.shape, dtype=torch.bool).to(past_key_values[0].device)
        mask = mask.scatter(-1, keep_idx, 1)
        
        # print(len(past_key_values))
        # print(past_key_values[0].shape, mask.shape, keep_idx.shape)

        k_hh_recent = past_key_values[0].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        v_hh_recent = past_key_values[1].squeeze()[mask].view(bsz, num_heads, -1, head_dim)

        self.hh_score= self.hh_score[mask].view(num_heads, self.k + self.recent_size)

        return (k_hh_recent, v_hh_recent)

    def _update_hh_score(self, attn_score_cache):

        num_new_tokens = attn_score_cache.shape[2]

        attn_score_cache = attn_score_cache.view(attn_score_cache.shape[0], self.group_size, -1, attn_score_cache.shape[2], attn_score_cache.shape[3])

        if self.hh_score is None:
            self.hh_score = attn_score_cache.sum(0).sum(0).sum(1)
        # elif num_new_tokens > self.cache_size:
        #     attn_score_cache = attn_score_cache.sum(0).sum(0).sum(1)
            
        else:
            # print(f"attn_score_cache:{attn_score_cache.shape}")
            # print(f"self.hh_score:{self.hh_score.shape}")
            attn_score_cache = attn_score_cache.sum(0).sum(0).sum(1)
            # print(f"attn_score_cache[:, :-num_new_tokens]:{attn_score_cache[:, :-num_new_tokens].shape}")
            attn_score_cache[:, :-num_new_tokens] += self.hh_score
            self.hh_score = attn_score_cache

    def _clean_scores(self):
        self.hh_score = None


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

        position_length = kv_seq_len
        if not position_ids.nelement() > 1:
            if position_length < position_ids.item()+1:
                position_length = position_ids.item()+1

        cos, sin = self.rotary_emb(value_states, seq_len=position_length)
        ### Shift Pos: query pos is min(cache_size, idx)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)
        
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
        causal_lm.to("cuda")
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

        position_length = kv_seq_len
        if not position_ids.nelement() > 1:
            if position_length < position_ids.item()+1:
                position_length = position_ids.item()+1

        cos, sin = self.rotary_emb(value_states, seq_len=position_length)
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
