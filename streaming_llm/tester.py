import time
import numpy as np

class Tester:
    def __init__(self):
        self.num_tokens = 0
        self.total_tokens = 0
        self.ppls = []
        self.elapsed_times = []
        self.token_counts = []

    def start(self):
        self.start_time = time.time()
        
    def step(self):
        self.num_tokens += 1
        
        if self.num_tokens % 100 == 0:
            self.stop()
            self.start()
    
    def record_ppl_and_print_stats(self, ppl):
        self.ppls.append((self.total_tokens, ppl))
        self.print_stats()

    def stop(self):
        if self.start_time is None:
            raise RuntimeError("先调用 start() 再调用 stop()")
        
        elapsed = time.time() - self.start_time
        self.elapsed_times.append(elapsed)
        self.token_counts.append(self.num_tokens)
        self.total_tokens += self.num_tokens
        self.num_tokens = 0
        self.start_time = None  # 重置计时器

    def get_stats(self):
        if not self.elapsed_times or self.total_tokens == 0:
            return {}
        
        total_time = sum(self.elapsed_times)
        avg_time = np.mean(self.elapsed_times)
        time_std = np.std(self.elapsed_times)
        throughput = self.total_tokens / total_time
        
        return {
            "total_runs": len(self.elapsed_times),
            "total_tokens": self.total_tokens,
            "total_time": total_time,
            "avg_time": avg_time,
            "time_std": time_std,
            "throughput": throughput,
            "per_token_time": total_time / self.total_tokens
        }

    def print_stats(self):
        stats = self.get_stats()
        if not stats:
            print("没有可用的统计数据")
            return
        
        print(f"\n{' 性能统计 ':=^40}")
        print(f"总测试次数: {stats['total_runs']}")
        print(f"总生成token数: {stats['total_tokens']}")
        print(f"总耗时: {stats['total_time']:.2f} 秒")
        print(f"平均耗时: {stats['avg_time']:.3f} ± {stats['time_std']:.3f} 秒")
        print(f"吞吐量: {stats['throughput']:.1f} tokens/秒")
        print(f"单token耗时: {stats['per_token_time'] * 1000:.2f} 毫秒")
        if len(self.ppls) > 0:
            print(f"本轮平均 ppl: {self.ppls[-1][1]:.2f}")
        print("=" * 40)
    
    def print_all_ppls(self):
        if len(self.ppls) > 0:
            print(f"\n{' ppl 统计 ':=^40}")
            for (num_tokens, ppl) in self.ppls:
                print(f"{num_tokens} tokens: {ppl:.2f}")
            print("=" * 40)
