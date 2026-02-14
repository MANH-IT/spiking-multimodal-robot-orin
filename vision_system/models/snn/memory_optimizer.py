"""
MemoryOptimizer skeleton cho Jetson AGX Orin.

Quản lý budget bộ nhớ logic ở mức Python; sau này có thể nối với
TensorRT hoặc các allocator tuỳ chỉnh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class MemoryOptimizer:
    """
    Quản lý bộ nhớ cho hệ thống real-time (skeleton).
    Kích thước hiện tại chỉ là placeholder, đơn vị byte.
    """

    budget: Dict[str, int] = field(
        default_factory=lambda: {
            "rgb_buffer": 1280 * 720 * 3 * 4,  # 10.5MB (float32)
            "depth_buffer": 640 * 480 * 2,  # 0.6MB (uint16)
            "model_weights": 256 * 1024 * 1024,  # 256MB
            "intermediate": 512 * 1024 * 1024,  # 512MB
        }
    )
    use_unified_memory: bool = True
    pinned_memory: bool = True

    def total_budget_mb(self) -> float:
        return sum(self.budget.values()) / (1024 * 1024)

    def set_component_budget(self, name: str, bytes_size: int) -> None:
        self.budget[name] = bytes_size

    def get_component_budget(self, name: str) -> int:
        return self.budget.get(name, 0)


__all__ = ["MemoryOptimizer"]

