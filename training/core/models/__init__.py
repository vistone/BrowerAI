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
    WebsiteIntentClassifier,
    CodeStyleAnalyzer,
    DependencyGraphLearner,
    DeviceAdaptationAnalyzer
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
    "WebsiteIntentClassifier",
    "CodeStyleAnalyzer",
    "DependencyGraphLearner",
    "DeviceAdaptationAnalyzer",
]
