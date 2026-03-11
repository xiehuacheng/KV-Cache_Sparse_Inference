import time
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class TimeCounter:
    _durations: np.ndarray = np.zeros(0, dtype=np.float64)
    _current_start: Optional[float] = None
    
    def start(self) -> None:
        """启动计时（自动结束未关闭的计时）"""
        if self._current_start is not None:
            self._add_duration(time.perf_counter() - self._current_start)
        self._current_start = time.perf_counter()
    
    def stop(self) -> float:
        """停止计时并返回本次耗时"""
        if self._current_start is None:
            raise RuntimeError("计时未启动")
        
        duration = time.perf_counter() - self._current_start
        self._add_duration(duration)
        self._current_start = None
        return duration
    
    def _add_duration(self, duration: float) -> None:
        """使用NumPy数组高效存储耗时数据"""
        self._durations = np.append(self._durations, duration)
    
    def print_statistics(self) -> None:
        """打印完整统计信息"""
        if len(self._durations) == 0:
            print("无计时记录")
            return
        
        print(f"【统计报告】\n"
              f"总记录数：{len(self._durations):,}\n"
              f"总耗时：{np.sum(self._durations):.4f}s\n"
              f"平均耗时：{np.mean(self._durations):.4f}s ± {np.std(self._durations):.4f}\n"
              f"最大耗时：{np.max(self._durations):.4f}s\n"
              f"最小耗时：{np.min(self._durations):.4f}s\n"
              f"中位数耗时：{np.median(self._durations):.4f}s")
    
    def reset(self) -> None:
        """重置统计"""
        self._durations = np.zeros(0, dtype=np.float64)
        self._current_start = None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()
