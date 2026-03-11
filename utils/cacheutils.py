from transformers.cache_utils import DynamicCache, SinkCache

class EditableDynamicCache(DynamicCache):
    def __init__(self):
        super().__init__()

    def replace(self, past_key_value, layer_idx):
        self.key_cache[layer_idx] = past_key_value[0]
        self.value_cache[layer_idx] = past_key_value[1]

class InitializedSinkCache(SinkCache):
    def __init__(self, window_length: int, num_sink_tokens: int):
        super().__init__(window_length, num_sink_tokens)
    
    def from_legacy_cache(self, past_key_values):
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
