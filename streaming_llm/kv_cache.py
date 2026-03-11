import torch


# 定义切片操作的函数，根据维度对输入张量进行切片
def slice2d(x, start, end):
    # 对x的第三维（k序列维度）进行切片操作
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    # 对x的第四维（v序列维度）进行切片操作
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    # 对x的第二维进行切片操作
    return x[:, start:end, ...]


# 定义一个字典，将维度映射到相应的切片函数
DIM_TO_SLICE = {
    1: slice1d,  # 1维切片操作
    2: slice2d,  # 2维切片操作
    3: slice3d,  # 3维切片操作
}


class StartRecentKVCache:
    # 该类用于实现Key-Value缓存，维护一个缓存策略（从开始和最近的序列数据获取）
    def __init__(
        self,
        start_size=4,  # 缓存的起始序列长度
        recent_size=512,  # 缓存的最近序列长度
        k_seq_dim=2,  # key序列所在的维度
        v_seq_dim=2,  # value序列所在的维度
    ):
        print(f"StartRecentKVCache: {start_size}, {recent_size}")
        # 初始化参数
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size  # 总缓存大小
        self.k_seq_dim = k_seq_dim  # key序列维度
        self.v_seq_dim = v_seq_dim  # value序列维度
        # 根据维度选择对应的切片函数
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

    def __call__(self, past_key_values):
        # 如果没有历史缓存（past_key_values 为 None），则返回 None
        if past_key_values is None:
            return None
        
        # 获取序列的长度
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        
        # 如果序列长度小于等于缓存大小，直接返回原始缓存
        if seq_len <= self.cache_size:
            return past_key_values
        
        # 如果序列长度超过缓存大小，则合并起始部分和最近的部分
        return [
            [
                # 连接起始部分和最近部分的key
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(k, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                # 连接起始部分和最近部分的value
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(v, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]
