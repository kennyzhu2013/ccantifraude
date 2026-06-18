"""重庆行业卡反诈语音质检 Agent.

基于 learn-claude-code 的 harness 工程哲学实现：
    Agent = Model(LLM) + Harness(工具 + 知识 + 观察 + 检索 + 记忆)

核心理念：智能来自模型，而不是堆叠 if-else 流程。本包提供的是让模型
在『重庆行业卡反诈质检』这一具体领域中工作的 harness（工具与知识环境）。
"""

from .schema import InspectionResult, RiskLevel
from .agent import QcAgent
from .config import Config

__all__ = ["InspectionResult", "RiskLevel", "QcAgent", "Config"]
__version__ = "0.1.0"
