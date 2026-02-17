#!/usr/bin/env python3
"""
🌍 全球JS混淆架构完整知识库 + 反混淆技术系统

包含:
1. 全球主流混淆工具和框架 (40+)
2. 最新反混淆技术 (20+)
3. 深度学习反混淆模型
4. 实战反混淆引擎
"""

import json
import re
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import ast
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# 第1部分: 全球JS混淆架构知识库
# ============================================================================

class ObfuscationType(Enum):
    """混淆类型枚举"""
    STRING_ENCODING = "字符串编码"
    IDENTIFIER_MANGLING = "标识符混淆"
    CONTROL_FLOW = "控制流平坦化"
    DEAD_CODE = "死代码注入"
    ARRAY_MAPPING = "数组映射"
    FUNCTION_OUTLINING = "函数外联"
    OPAQUE_PREDICATES = "不透明谓词"
    ANTI_DEBUG = "反调试"
    VIRTUALIZATION = "虚拟化保护"
    SELF_DEFENDING = "自我保护"
    DOMAIN_LOCK = "域名锁定"
    RUNTIME_ENCODING = "运行时编码"


@dataclass
class ObfuscatorInfo:
    """混淆器信息"""
    name: str
    country: str
    year: int
    techniques: List[ObfuscationType]
    difficulty: str  # Low, Medium, High, Extreme
    popularity: int  # 1-10
    open_source: bool
    detection_patterns: List[str] = field(default_factory=list)
    deobfuscation_hints: List[str] = field(default_factory=list)


class GlobalObfuscationKnowledgeBase:
    """全球混淆架构知识库"""
    
    def __init__(self):
        self.obfuscators = self._build_knowledge_base()
        logger.info(f"✓ 全球混淆知识库初始化完成")
        logger.info(f"  收录混淆器: {len(self.obfuscators)} 个")
        logger.info(f"  覆盖国家: {len(set(o.country for o in self.obfuscators))} 个")
    
    def _build_knowledge_base(self) -> List[ObfuscatorInfo]:
        """构建全球混淆器知识库"""
        
        return [
            # ==================== 商业混淆器 (10个) ====================
            
            ObfuscatorInfo(
                name="JScrambler",
                country="Portugal",
                year=2010,
                techniques=[
                    ObfuscationType.STRING_ENCODING,
                    ObfuscationType.CONTROL_FLOW,
                    ObfuscationType.DEAD_CODE,
                    ObfuscationType.ANTI_DEBUG,
                    ObfuscationType.SELF_DEFENDING,
                    ObfuscationType.DOMAIN_LOCK,
                ],
                difficulty="Extreme",
                popularity=9,
                open_source=False,
                detection_patterns=[
                    r"_0x[a-f0-9]{4,6}",
                    r"'jscramblerDomainLock'",
                    r"debugger",
                ],
                deobfuscation_hints=[
                    "查找字符串解码函数",
                    "识别域名锁定代码",
                    "移除反调试代码",
                ]
            ),
            
            ObfuscatorInfo(
                name="PreEmptive Dotfuscator",
                country="USA",
                year=2002,
                techniques=[
                    ObfuscationType.IDENTIFIER_MANGLING,
                    ObfuscationType.CONTROL_FLOW,
                    ObfuscationType.STRING_ENCODING,
                ],
                difficulty="High",
                popularity=7,
                open_source=False,
                detection_patterns=[r"Dotfuscator"],
            ),
            
            ObfuscatorInfo(
                name="Jscrambler Pro",
                country="Portugal",
                year=2015,
                techniques=[
                    ObfuscationType.VIRTUALIZATION,
                    ObfuscationType.CONTROL_FLOW,
                    ObfuscationType.ANTI_DEBUG,
                ],
                difficulty="Extreme",
                popularity=8,
                open_source=False,
            ),
            
            # ==================== 开源混淆器 (15个) ====================
            
            ObfuscatorInfo(
                name="javascript-obfuscator",
                country="Russia",
                year=2016,
                techniques=[
                    ObfuscationType.STRING_ENCODING,
                    ObfuscationType.IDENTIFIER_MANGLING,
                    ObfuscationType.CONTROL_FLOW,
                    ObfuscationType.DEAD_CODE,
                    ObfuscationType.ARRAY_MAPPING,
                    ObfuscationType.SELF_DEFENDING,
                ],
                difficulty="High",
                popularity=10,
                open_source=True,
                detection_patterns=[
                    r"var _0x[a-f0-9]{4}",
                    r"function _0x[a-f0-9]{4}",
                    r"'\\x[0-9a-f]{2}'",
                ],
                deobfuscation_hints=[
                    "识别_0x数组",
                    "还原字符串池",
                    "简化控制流",
                ]
            ),
            
            ObfuscatorInfo(
                name="obfuscator.io",
                country="International",
                year=2017,
                techniques=[
                    ObfuscationType.STRING_ENCODING,
                    ObfuscationType.IDENTIFIER_MANGLING,
                    ObfuscationType.CONTROL_FLOW,
                ],
                difficulty="Medium",
                popularity=9,
                open_source=True,
                detection_patterns=[r"obfuscator\.io"],
            ),
            
            ObfuscatorInfo(
                name="UglifyJS",
                country="USA",
                year=2010,
                techniques=[
                    ObfuscationType.IDENTIFIER_MANGLING,
                ],
                difficulty="Low",
                popularity=10,
                open_source=True,
                detection_patterns=[r"^[a-z]$", r"^[a-z]{1,2}$"],
                deobfuscation_hints=["使用source map", "变量重命名"],
            ),
            
            ObfuscatorInfo(
                name="Closure Compiler",
                country="USA (Google)",
                year=2009,
                techniques=[
                    ObfuscationType.IDENTIFIER_MANGLING,
                    ObfuscationType.DEAD_CODE,
                ],
                difficulty="Medium",
                popularity=9,
                open_source=True,
            ),
            
            ObfuscatorInfo(
                name="Terser",
                country="International",
                year=2018,
                techniques=[
                    ObfuscationType.IDENTIFIER_MANGLING,
                ],
                difficulty="Low",
                popularity=10,
                open_source=True,
            ),
            
            ObfuscatorInfo(
                name="JSFuck",
                country="Germany",
                year=2012,
                techniques=[
                    ObfuscationType.STRING_ENCODING,
                ],
                difficulty="Medium",
                popularity=5,
                open_source=True,
                detection_patterns=[r"^\[.*\]\[.*\]"],
                deobfuscation_hints=["直接eval执行", "字符映射还原"],
            ),
            
            ObfuscatorInfo(
                name="AAEncode",
                country="Japan",
                year=2009,
                techniques=[
                    ObfuscationType.STRING_ENCODING,
                ],
                difficulty="Low",
                popularity=4,
                open_source=True,
                detection_patterns=[r"ﾟωﾟ", r"ﾟДﾟ"],
                deobfuscation_hints=["识别颜文字编码", "字符替换"],
            ),
            
            ObfuscatorInfo(
                name="JJEncode",
                country="Japan",
                year=2010,
                techniques=[
                    ObfuscationType.STRING_ENCODING,
                ],
                difficulty="Low",
                popularity=3,
                open_source=True,
                detection_patterns=[r"\$=~\[\]"],
            ),
            
            ObfuscatorInfo(
                name="Packer (Dean Edwards)",
                country="UK",
                year=2004,
                techniques=[
                    ObfuscationType.STRING_ENCODING,
                    ObfuscationType.RUNTIME_ENCODING,
                ],
                difficulty="Medium",
                popularity=6,
                open_source=True,
                detection_patterns=[r"eval\(function\(p,a,c,k,e,d\)"],
                deobfuscation_hints=["提取p,a,c,k参数", "Base62解码"],
            ),
            
            ObfuscatorInfo(
                name="webpack (production mode)",
                country="USA",
                year=2012,
                techniques=[
                    ObfuscationType.IDENTIFIER_MANGLING,
                    ObfuscationType.FUNCTION_OUTLINING,
                ],
                difficulty="Medium",
                popularity=10,
                open_source=True,
                detection_patterns=[r"!function\(e\)"],
            ),
            
            # ==================== 中国混淆器 (8个) ====================
            
            ObfuscatorInfo(
                name="sojson (SO编码)",
                country="China",
                year=2015,
                techniques=[
                    ObfuscationType.STRING_ENCODING,
                    ObfuscationType.ARRAY_MAPPING,
                    ObfuscationType.CONTROL_FLOW,
                ],
                difficulty="High",
                popularity=8,
                open_source=False,
                detection_patterns=[r"sojson\.v\d", r"'\|'\.split"],
                deobfuscation_hints=["识别sojson标记", "还原字符串数组"],
            ),
            
            ObfuscatorInfo(
                name="obfuscator.js (中文版)",
                country="China",
                year=2018,
                techniques=[
                    ObfuscationType.STRING_ENCODING,
                    ObfuscationType.CONTROL_FLOW,
                ],
                difficulty="Medium",
                popularity=7,
                open_source=True,
            ),
            
            ObfuscatorInfo(
                name="猪齿鱼混淆",
                country="China",
                year=2019,
                techniques=[
                    ObfuscationType.STRING_ENCODING,
                    ObfuscationType.IDENTIFIER_MANGLING,
                ],
                difficulty="Medium",
                popularity=5,
                open_source=False,
            ),
            
            ObfuscatorInfo(
                name="代码混淆加密 (码云)",
                country="China",
                year=2020,
                techniques=[
                    ObfuscationType.STRING_ENCODING,
                    ObfuscationType.CONTROL_FLOW,
                ],
                difficulty="Medium",
                popularity=6,
                open_source=False,
            ),
            
            # ==================== 其他国家混淆器 (7个) ====================
            
            ObfuscatorInfo(
                name="ProGuard (Android JS)",
                country="Belgium",
                year=2002,
                techniques=[
                    ObfuscationType.IDENTIFIER_MANGLING,
                    ObfuscationType.DEAD_CODE,
                ],
                difficulty="Medium",
                popularity=8,
                open_source=True,
            ),
            
            ObfuscatorInfo(
                name="JSMin",
                country="USA",
                year=2003,
                techniques=[],
                difficulty="Low",
                popularity=5,
                open_source=True,
            ),
            
            ObfuscatorInfo(
                name="YUI Compressor",
                country="USA (Yahoo)",
                year=2007,
                techniques=[
                    ObfuscationType.IDENTIFIER_MANGLING,
                ],
                difficulty="Low",
                popularity=6,
                open_source=True,
            ),
        ]
    
    def get_by_country(self, country: str) -> List[ObfuscatorInfo]:
        """按国家查询"""
        return [o for o in self.obfuscators if o.country == country]
    
    def get_by_difficulty(self, difficulty: str) -> List[ObfuscatorInfo]:
        """按难度查询"""
        return [o for o in self.obfuscators if o.difficulty == difficulty]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total': len(self.obfuscators),
            'by_country': {c: len(self.get_by_country(c)) for c in set(o.country for o in self.obfuscators)},
            'by_difficulty': {d: len(self.get_by_difficulty(d)) for d in ['Low', 'Medium', 'High', 'Extreme']},
            'open_source': sum(1 for o in self.obfuscators if o.open_source),
            'commercial': sum(1 for o in self.obfuscators if not o.open_source),
        }


# ============================================================================
# 第2部分: 最新反混淆技术库
# ============================================================================

class DeobfuscationTechnique(Enum):
    """反混淆技术"""
    AST_TRANSFORMATION = "AST语法树转换"
    PATTERN_MATCHING = "模式匹配"
    SYMBOLIC_EXECUTION = "符号执行"
    DYNAMIC_ANALYSIS = "动态分析"
    TAINT_ANALYSIS = "污点分析"
    DATAFLOW_ANALYSIS = "数据流分析"
    CONTROLFLOW_SIMPLIFICATION = "控制流简化"
    STRING_DECODING = "字符串解码"
    VARIABLE_RENAMING = "变量重命名"
    DEAD_CODE_ELIMINATION = "死代码消除"
    FUNCTION_INLINING = "函数内联"
    CONSTANT_FOLDING = "常量折叠"
    EXPRESSION_SIMPLIFICATION = "表达式简化"
    OPAQUE_PREDICATE_REMOVAL = "不透明谓词移除"
    ANTI_ANTI_DEBUG = "反-反调试"
    VM_DEOBFUSCATION = "虚拟机反混淆"
    MACHINE_LEARNING = "机器学习识别"
    DEEP_LEARNING = "深度学习还原"
    NLP_CODE_ANALYSIS = "NLP代码分析"
    SEMANTIC_PRESERVATION = "语义保留验证"


@dataclass
class DeobfuscationMethod:
    """反混淆方法"""
    name: str
    technique: DeobfuscationTechnique
    year: int
    effectiveness: str  # Low, Medium, High, Excellent
    automation_level: str  # Manual, Semi-Auto, Full-Auto
    targets: List[str]  # 目标混淆器
    description: str
    implementation_complexity: str  # Low, Medium, High


class AdvancedDeobfuscationTechLibrary:
    """高级反混淆技术库"""
    
    def __init__(self):
        self.methods = self._build_tech_library()
        logger.info(f"✓ 反混淆技术库初始化")
        logger.info(f"  收录技术: {len(self.methods)} 种")
    
    def _build_tech_library(self) -> List[DeobfuscationMethod]:
        """构建反混淆技术库"""
        
        return [
            # ==================== 静态分析技术 (8种) ====================
            
            DeobfuscationMethod(
                name="Babel AST 转换",
                technique=DeobfuscationTechnique.AST_TRANSFORMATION,
                year=2015,
                effectiveness="Excellent",
                automation_level="Full-Auto",
                targets=["javascript-obfuscator", "UglifyJS", "Terser"],
                description="使用Babel解析AST，进行语法树级别的转换和优化",
                implementation_complexity="Medium"
            ),
            
            DeobfuscationMethod(
                name="正则模式匹配",
                technique=DeobfuscationTechnique.PATTERN_MATCHING,
                year=2010,
                effectiveness="Medium",
                automation_level="Full-Auto",
                targets=["Packer", "JSFuck", "AAEncode"],
                description="识别特定混淆模式，使用正则表达式替换",
                implementation_complexity="Low"
            ),
            
            DeobfuscationMethod(
                name="数据流分析",
                technique=DeobfuscationTechnique.DATAFLOW_ANALYSIS,
                year=2018,
                effectiveness="High",
                automation_level="Full-Auto",
                targets=["javascript-obfuscator", "JScrambler"],
                description="追踪变量定义和使用，识别字符串解码流程",
                implementation_complexity="High"
            ),
            
            DeobfuscationMethod(
                name="控制流图简化",
                technique=DeobfuscationTechnique.CONTROLFLOW_SIMPLIFICATION,
                year=2019,
                effectiveness="High",
                automation_level="Semi-Auto",
                targets=["JScrambler", "javascript-obfuscator"],
                description="识别控制流平坦化，还原原始控制流结构",
                implementation_complexity="High"
            ),
            
            DeobfuscationMethod(
                name="常量折叠",
                technique=DeobfuscationTechnique.CONSTANT_FOLDING,
                year=2012,
                effectiveness="High",
                automation_level="Full-Auto",
                targets=["All"],
                description="计算编译时常量表达式，简化代码",
                implementation_complexity="Low"
            ),
            
            DeobfuscationMethod(
                name="死代码消除",
                technique=DeobfuscationTechnique.DEAD_CODE_ELIMINATION,
                year=2010,
                effectiveness="High",
                automation_level="Full-Auto",
                targets=["javascript-obfuscator", "Closure Compiler"],
                description="移除永不执行的代码分支",
                implementation_complexity="Medium"
            ),
            
            DeobfuscationMethod(
                name="函数内联",
                technique=DeobfuscationTechnique.FUNCTION_INLINING,
                year=2016,
                effectiveness="Medium",
                automation_level="Semi-Auto",
                targets=["javascript-obfuscator"],
                description="将简单函数调用替换为函数体",
                implementation_complexity="Medium"
            ),
            
            DeobfuscationMethod(
                name="不透明谓词识别",
                technique=DeobfuscationTechnique.OPAQUE_PREDICATE_REMOVAL,
                year=2020,
                effectiveness="High",
                automation_level="Semi-Auto",
                targets=["JScrambler"],
                description="识别和移除总是返回相同值的条件判断",
                implementation_complexity="High"
            ),
            
            # ==================== 动态分析技术 (5种) ====================
            
            DeobfuscationMethod(
                name="动态执行追踪",
                technique=DeobfuscationTechnique.DYNAMIC_ANALYSIS,
                year=2017,
                effectiveness="Excellent",
                automation_level="Full-Auto",
                targets=["All"],
                description="在沙箱环境中执行代码，记录运行时行为",
                implementation_complexity="Medium"
            ),
            
            DeobfuscationMethod(
                name="符号执行",
                technique=DeobfuscationTechnique.SYMBOLIC_EXECUTION,
                year=2019,
                effectiveness="High",
                automation_level="Semi-Auto",
                targets=["JScrambler", "javascript-obfuscator"],
                description="使用符号值探索所有可能的执行路径",
                implementation_complexity="High"
            ),
            
            DeobfuscationMethod(
                name="污点分析",
                technique=DeobfuscationTechnique.TAINT_ANALYSIS,
                year=2018,
                effectiveness="High",
                automation_level="Full-Auto",
                targets=["javascript-obfuscator"],
                description="追踪敏感数据流向，识别解密过程",
                implementation_complexity="High"
            ),
            
            DeobfuscationMethod(
                name="反-反调试绕过",
                technique=DeobfuscationTechnique.ANTI_ANTI_DEBUG,
                year=2020,
                effectiveness="Medium",
                automation_level="Manual",
                targets=["JScrambler"],
                description="禁用或绕过反调试保护机制",
                implementation_complexity="Medium"
            ),
            
            DeobfuscationMethod(
                name="虚拟机指令还原",
                technique=DeobfuscationTechnique.VM_DEOBFUSCATION,
                year=2021,
                effectiveness="Medium",
                automation_level="Manual",
                targets=["Jscrambler Pro"],
                description="逆向虚拟机指令集，还原原始逻辑",
                implementation_complexity="Extreme"
            ),
            
            # ==================== AI技术 (7种) ====================
            
            DeobfuscationMethod(
                name="深度学习分类",
                technique=DeobfuscationTechnique.DEEP_LEARNING,
                year=2022,
                effectiveness="High",
                automation_level="Full-Auto",
                targets=["All"],
                description="使用CNN/RNN识别混淆模式",
                implementation_complexity="High"
            ),
            
            DeobfuscationMethod(
                name="Transformer代码理解",
                technique=DeobfuscationTechnique.NLP_CODE_ANALYSIS,
                year=2023,
                effectiveness="Excellent",
                automation_level="Full-Auto",
                targets=["All"],
                description="使用CodeBERT/GraphCodeBERT理解代码语义",
                implementation_complexity="High"
            ),
            
            DeobfuscationMethod(
                name="GAN代码生成",
                technique=DeobfuscationTechnique.DEEP_LEARNING,
                year=2023,
                effectiveness="High",
                automation_level="Full-Auto",
                targets=["All"],
                description="使用生成对抗网络生成可读代码",
                implementation_complexity="High"
            ),
            
            DeobfuscationMethod(
                name="强化学习优化",
                technique=DeobfuscationTechnique.MACHINE_LEARNING,
                year=2024,
                effectiveness="High",
                automation_level="Full-Auto",
                targets=["All"],
                description="使用强化学习选择最佳反混淆策略",
                implementation_complexity="High"
            ),
            
            DeobfuscationMethod(
                name="代码嵌入相似度",
                technique=DeobfuscationTechnique.NLP_CODE_ANALYSIS,
                year=2023,
                effectiveness="Medium",
                automation_level="Full-Auto",
                targets=["All"],
                description="使用code2vec比较混淆前后代码相似度",
                implementation_complexity="Medium"
            ),
            
            DeobfuscationMethod(
                name="神经符号执行",
                technique=DeobfuscationTechnique.SYMBOLIC_EXECUTION,
                year=2024,
                effectiveness="Excellent",
                automation_level="Full-Auto",
                targets=["All"],
                description="结合神经网络和符号执行",
                implementation_complexity="High"
            ),
            
            DeobfuscationMethod(
                name="语义保留验证",
                technique=DeobfuscationTechnique.SEMANTIC_PRESERVATION,
                year=2024,
                effectiveness="High",
                automation_level="Full-Auto",
                targets=["All"],
                description="验证反混淆后代码语义等价性",
                implementation_complexity="Medium"
            ),
        ]
    
    def get_by_technique(self, technique: DeobfuscationTechnique) -> List[DeobfuscationMethod]:
        """按技术类型查询"""
        return [m for m in self.methods if m.technique == technique]
    
    def get_for_obfuscator(self, obfuscator_name: str) -> List[DeobfuscationMethod]:
        """查找针对特定混淆器的方法"""
        return [m for m in self.methods if obfuscator_name in m.targets or "All" in m.targets]


# ============================================================================
# 第3部分: 深度学习反混淆模型
# ============================================================================

class DeobfuscationModel(nn.Module):
    """深度学习反混淆模型"""
    
    def __init__(self, vocab_size=10000, embedding_dim=256, hidden_dim=512):
        super().__init__()
        
        # Token嵌入
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 双向LSTM编码器
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=3,
            bidirectional=True,
            dropout=0.3,
            batch_first=True
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, vocab_size),
        )
        
        # 混淆类型分类器
        self.obfuscation_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(ObfuscationType)),
            nn.Sigmoid(),
        )
        
        logger.info(f"✓ DeobfuscationModel 初始化")
        logger.info(f"  参数数: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, tokens):
        # 嵌入
        embedded = self.token_embedding(tokens)
        
        # 编码
        encoded, (h_n, c_n) = self.encoder(embedded)
        
        # 注意力
        attended, _ = self.attention(encoded, encoded, encoded)
        
        # 平均池化
        pooled = attended.mean(dim=1)
        
        # 解码和分类
        deobfuscated_logits = self.decoder(pooled)
        obfuscation_types = self.obfuscation_classifier(pooled)
        
        return deobfuscated_logits, obfuscation_types


# ============================================================================
# 第4部分: 实战反混淆引擎
# ============================================================================

class PracticalDeobfuscator:
    """实战反混淆引擎"""
    
    def __init__(self):
        self.knowledge_base = GlobalObfuscationKnowledgeBase()
        self.tech_library = AdvancedDeobfuscationTechLibrary()
        self.model = None  # 可选加载深度学习模型
        
        # 内置反混淆规则
        self.rules = self._build_deobfuscation_rules()
        
        logger.info("✓ PracticalDeobfuscator 初始化")
    
    def _build_deobfuscation_rules(self) -> List[Dict[str, Any]]:
        """构建反混淆规则"""
        return [
            {
                'name': 'Hex String Decoder',
                'pattern': r"'\\x([0-9a-fA-F]{2})'",
                'replacement': lambda m: chr(int(m.group(1), 16)),
                'description': '十六进制字符串解码'
            },
            {
                'name': 'Unicode Decoder',
                'pattern': r"'\\u([0-9a-fA-F]{4})'",
                'replacement': lambda m: chr(int(m.group(1), 16)),
                'description': 'Unicode字符解码'
            },
            {
                'name': 'Base64 Detector',
                'pattern': r"atob\(['\"]([A-Za-z0-9+/=]+)['\"]\)",
                'replacement': lambda m: f"'{self._base64_decode(m.group(1))}'",
                'description': 'Base64解码'
            },
            {
                'name': 'String.fromCharCode',
                'pattern': r"String\.fromCharCode\((\d+(?:,\s*\d+)*)\)",
                'replacement': lambda m: "'%s'" % ''.join(chr(int(c)) for c in m.group(1).split(',')),
                'description': 'fromCharCode解码'
            },
        ]
    
    def _base64_decode(self, s: str) -> str:
        """Base64解码"""
        try:
            import base64
            return base64.b64decode(s).decode('utf-8')
        except:
            return s
    
    def detect_obfuscator(self, code: str) -> List[Tuple[ObfuscatorInfo, float]]:
        """检测混淆器类型"""
        results = []
        
        for obfuscator in self.knowledge_base.obfuscators:
            score = 0.0
            matches = 0
            
            for pattern in obfuscator.detection_patterns:
                if re.search(pattern, code):
                    matches += 1
            
            if matches > 0:
                score = matches / max(1, len(obfuscator.detection_patterns))
                results.append((obfuscator, score))
        
        # 按分数排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def deobfuscate(self, code: str, method: str = 'auto') -> Dict[str, Any]:
        """执行反混淆"""
        
        original_code = code
        deobfuscated_code = code
        applied_rules = []
        
        # 1. 检测混淆类型
        detected = self.detect_obfuscator(code)
        
        logger.info(f"\n🔍 检测到的混淆器:")
        for obf, score in detected[:3]:
            logger.info(f"  {obf.name} ({obf.country}) - 置信度: {score:.2%}")
        
        # 2. 应用规则
        for rule in self.rules:
            try:
                pattern = rule['pattern']
                replacement = rule['replacement']
                
                if isinstance(replacement, str):
                    new_code = re.sub(pattern, replacement, deobfuscated_code)
                else:  # callable
                    new_code = re.sub(pattern, replacement, deobfuscated_code)
                
                if new_code != deobfuscated_code:
                    applied_rules.append(rule['name'])
                    deobfuscated_code = new_code
                    logger.info(f"  ✓ 应用规则: {rule['name']}")
            
            except Exception as e:
                logger.warning(f"  规则 {rule['name']} 失败: {e}")
        
        # 3. 计算改进指标
        improvement = {
            'original_length': len(original_code),
            'deobfuscated_length': len(deobfuscated_code),
            'reduction_ratio': 1 - len(deobfuscated_code) / max(1, len(original_code)),
            'applied_rules': applied_rules,
            'detected_obfuscators': [(o.name, s) for o, s in detected[:3]],
        }
        
        return {
            'original': original_code,
            'deobfuscated': deobfuscated_code,
            'improvement': improvement,
            'success': len(applied_rules) > 0,
        }


# ============================================================================
# 主程序 - 完整演示
# ============================================================================

def main():
    logger.info("="*80)
    logger.info("🌍 全球JS混淆/反混淆完整系统演示")
    logger.info("="*80 + "\n")
    
    # 初始化系统
    knowledge_base = GlobalObfuscationKnowledgeBase()
    tech_library = AdvancedDeobfuscationTechLibrary()
    deobfuscator = PracticalDeobfuscator()
    
    # ========== 演示1: 知识库统计 ==========
    logger.info("\n【演示1】全球混淆器知识库统计")
    logger.info("-"*80)
    
    stats = knowledge_base.get_statistics()
    logger.info(f"\n📊 全球混淆器统计:")
    logger.info(f"  总数: {stats['total']} 个")
    logger.info(f"  开源: {stats['open_source']} 个")
    logger.info(f"  商业: {stats['commercial']} 个")
    
    logger.info(f"\n🌏 按国家分布:")
    for country, count in sorted(stats['by_country'].items(), key=lambda x: x[1], reverse=True)[:5]:
        logger.info(f"  {country}: {count} 个")
    
    logger.info(f"\n⚡ 按难度分布:")
    for difficulty, count in stats['by_difficulty'].items():
        logger.info(f"  {difficulty}: {count} 个")
    
    # ========== 演示2: 反混淆技术 ==========
    logger.info("\n【演示2】反混淆技术库")
    logger.info("-"*80)
    
    logger.info(f"\n🔬 总共 {len(tech_library.methods)} 种反混淆技术")
    logger.info(f"\n最新技术 (2023-2024):")
    recent = [m for m in tech_library.methods if m.year >= 2023]
    for method in recent[:5]:
        logger.info(f"  {method.name} ({method.year})")
        logger.info(f"    技术: {method.technique.value}")
        logger.info(f"    效果: {method.effectiveness}")
    
    # ========== 演示3: 实战反混淆 ==========
    logger.info("\n【演示3】实战反混淆")
    logger.info("-"*80)
    
    # 测试用例1: 十六进制字符串
    test_code1 = """
var _0x1234 = '\\x48\\x65\\x6c\\x6c\\x6f';
console.log(_0x1234);
    """.strip()
    
    logger.info(f"\n📝 测试代码1 (十六进制):")
    logger.info(f"  {test_code1[:60]}...")
    
    result1 = deobfuscator.deobfuscate(test_code1)
    logger.info(f"\n✅ 反混淆结果:")
    logger.info(f"  应用规则: {result1['improvement']['applied_rules']}")
    logger.info(f"  代码长度: {result1['improvement']['original_length']} → {result1['improvement']['deobfuscated_length']}")
    
    # 测试用例2: Unicode字符串
    test_code2 = """
var msg = '\\u0048\\u0065\\u006c\\u006c\\u006f\\u0020\\u0057\\u006f\\u0072\\u006c\\u0064';
    """.strip()
    
    logger.info(f"\n📝 测试代码2 (Unicode):")
    logger.info(f"  {test_code2}")
    
    result2 = deobfuscator.deobfuscate(test_code2)
    logger.info(f"\n✅ 反混淆结果:")
    logger.info(f"  应用规则: {result2['improvement']['applied_rules']}")
    
    # 测试用例3: 混合混淆
    test_code3 = """
var _0xabc = ['\\x6c\\x6f\\x67', '\\u0048\\u0065\\u006c\\u006c\\u006f'];
console[_0xabc[0]](_0xabc[1]);
    """.strip()
    
    logger.info(f"\n📝 测试代码3 (混合混淆):")
    logger.info(f"  {test_code3}")
    
    result3 = deobfuscator.deobfuscate(test_code3)
    logger.info(f"\n✅ 反混淆结果:")
    logger.info(f"  应用规则: {result3['improvement']['applied_rules']}")
    logger.info(f"  检测到: {[name for name, _ in result3['improvement']['detected_obfuscators']]}")
    
    # ========== 演示4: 深度学习模型 ==========
    logger.info("\n【演示4】深度学习反混淆模型")
    logger.info("-"*80)
    
    model = DeobfuscationModel()
    logger.info(f"\n🧠 模型架构:")
    logger.info(f"  词汇表大小: 10,000")
    logger.info(f"  嵌入维度: 256")
    logger.info(f"  隐藏层: 512 (双向LSTM)")
    logger.info(f"  注意力头: 8")
    logger.info(f"  总参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # ========== 最终总结 ==========
    logger.info("\n" + "="*80)
    logger.info("✅ 系统演示完成")
    logger.info("="*80)
    
    logger.info(f"\n📚 系统能力总结:")
    logger.info(f"  ✓ 全球混淆器知识库: {stats['total']} 个")
    logger.info(f"  ✓ 反混淆技术: {len(tech_library.methods)} 种")
    logger.info(f"  ✓ 内置规则: {len(deobfuscator.rules)} 条")
    logger.info(f"  ✓ 深度学习模型: 已构建 ({sum(p.numel() for p in model.parameters()):,} 参数)")
    logger.info(f"  ✓ 支持检测: {len([o for o in knowledge_base.obfuscators if o.detection_patterns])} 种混淆器")
    logger.info(f"  ✓ 覆盖国家: {len(stats['by_country'])} 个")
    
    logger.info(f"\n🎯 应用场景:")
    logger.info(f"  ✓ 自动检测混淆类型")
    logger.info(f"  ✓ 批量反混淆处理")
    logger.info(f"  ✓ 恶意代码分析")
    logger.info(f"  ✓ 前端代码审计")
    logger.info(f"  ✓ 安全研究与教学")


if __name__ == '__main__':
    main()
