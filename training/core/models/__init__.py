"""
Neural Network Architectures for BrowerAI

Modern model architectures following best practices from:
- GPT (decoder-only transformers)
- BERT (encoder-only transformers)
- T5 (encoder-decoder transformers)
- Code generation models (CodeGen, Codex principles)
- Multi-modal learning (CLIP, ALIGN)
- Graph neural networks for dependencies
"""

from .base import BaseModel
from .transformer import TransformerEncoder, TransformerDecoder, TransformerSeq2Seq
from .attention import MultiHeadAttention, PositionalEncoding
from .parsers import HTMLParser, CSSParser, JSParser
from .deobfuscator import CodeDeobfuscator
from .website_learner import (
    HolisticWebsiteLearner,
    CodeStyleAnalyzer,
    DependencyGraphLearner,
    DeviceAdaptationAnalyzer
)
from .website_generator import (
    WebsiteGenerator,
    CodeEncoder,
    CodeDecoder,
    WebsiteIntentClassifier
)

__all__ = [
    "BaseModel",
    "TransformerEncoder",
    "TransformerDecoder", 
    "TransformerSeq2Seq",
    "MultiHeadAttention",
    "PositionalEncoding",
    "HTMLParser",
    "CSSParser",
    "JSParser",
    "CodeDeobfuscator",
    "HolisticWebsiteLearner",
    "CodeStyleAnalyzer",
    "DependencyGraphLearner",
    "DeviceAdaptationAnalyzer",
    # 端到端网站生成（新增）
    "WebsiteGenerator",
    "CodeEncoder",
    "CodeDecoder",
    "WebsiteIntentClassifier",
]
