#!/usr/bin/env python3
"""
混合框架检测系统
策略: 规则检测 (快速、准确) + AI预测 (兜底)
"""

import re
import onnxruntime as ort
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class DetectionMethod(Enum):
    RULE_BASED = "rule_based"
    AI_BASED = "ai_based"
    HYBRID = "hybrid"


@dataclass
class DetectionResult:
    """检测结果数据类"""
    framework: str
    confidence: float
    method: DetectionMethod
    details: Dict = None


class RuleBasedDetector:
    """基于规则的框架检测（快速、准确）"""
    
    def __init__(self):
        self.rules = {
            # 前端框架
            "react": [
                r"from\s+['\"]react['\"]",
                r"import\s+React\b",
                r"React\.createElement",
                r"useEffect|useState|useContext",
                r"ReactDOM\.(render|createRoot)",
            ],
            "vue": [
                r"from\s+['\"]vue['\"]",
                r"import\s+\{.*Vue\b",
                r"Vue\.createApp",
                r"<template>",
                r"defineComponent|useCompositionAPI",
            ],
            "angular": [
                r"from\s+['\"]@angular",
                r"import\s+\{.*Component\b.*\}\s+from\s+['\"]@angular",
                r"@Component|@NgModule|@Injectable",
                r"CommonModule|FormsModule",
            ],
            "svelte": [
                r"from\s+['\"]svelte['\"]",
                r"import\s+{.*}\s+from\s+['\"]svelte['\"]",
                r"<script>|{@html|{@const}",
            ],
            "solid": [
                r"from\s+['\"]solid-js['\"]",
                r"createSignal|createEffect|createResource",
            ],
            "preact": [
                r"from\s+['\"]preact['\"]",
                r"h\s+\(",
            ],
            "nextjs": [
                r"from\s+['\"]next/",
                r"getServerSideProps|getStaticProps",
                r"useRouter.*from\s+['\"]next/router['\"]",
            ],
            "nuxt": [
                r"from\s+['\"]nuxt['\"]",
                r"nuxt\.config",
            ],
            "gatsby": [
                r"gatsby-config\.js",
                r"gatsby-node\.js",
                r"gatsby-browser\.js",
            ],
            "remix": [
                r"from\s+['\"]@remix-run",
                r"loader|action",
            ],
            
            # 后端框架
            "express": [
                r"require\s*\(\s*['\"]express['\"]",
                r"from\s+['\"]express['\"]",
                r"express\.Router|app\.get|app\.post",
                r"Express\(\)",
            ],
            "koa": [
                r"require\s*\(\s*['\"]koa['\"]",
                r"from\s+['\"]koa['\"]",
                r"Koa\(\)|ctx\.body",
            ],
            "fastify": [
                r"require\s*\(\s*['\"]fastify['\"]",
                r"from\s+['\"]fastify['\"]",
                r"fastify\(\)",
            ],
            "hapi": [
                r"require\s*\(\s*['\"]@hapi",
                r"Hapi\.Server",
            ],
            "nest": [
                r"from\s+['\"]@nestjs",
                r"@Module|@Controller|@Service",
            ],
            
            # 工具库
            "lodash": [
                r"require\s*\(\s*['\"]lodash['\"]",
                r"from\s+['\"]lodash['\"]",
                r"_\.(\w+)\(",
                r"import\s+_\s+from",
            ],
            "ramda": [
                r"require\s*\(\s*['\"]ramda['\"]",
                r"from\s+['\"]ramda['\"]",
                r"R\.(\w+)\(",
            ],
            "underscore": [
                r"require\s*\(\s*['\"]underscore['\"]",
                r"from\s+['\"]underscore['\"]",
            ],
            
            # 其他
            "alpine": [
                r"alpine\.js",
                r"x-data|x-init|x-show",
            ],
            "lit": [
                r"from\s+['\"]lit['\"]",
                r"LitElement|html\`",
            ],
            "stencil": [
                r"from\s+['\"]@stencil",
                r"@Component|@Prop",
            ],
            "ember": [
                r"from\s+['\"]ember",
                r"@ember/",
            ],
            "backbone": [
                r"require\s*\(\s*['\"]backbone['\"]",
                r"Backbone\.(Model|View|Router)",
            ],
            "knockout": [
                r"require\s*\(\s*['\"]knockout['\"]",
                r"ko\.observable|ko\.computed",
            ],
        }
        
        # 编译正则表达式以提升性能
        self.compiled_rules = {}
        for framework, patterns in self.rules.items():
            self.compiled_rules[framework] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                for pattern in patterns
            ]
    
    def detect(self, code: str) -> Optional[Tuple[str, float, List[str]]]:
        """
        基于规则检测框架
        
        Returns:
            (framework, confidence, matched_patterns) 或 None
        """
        matches = {}
        matched_patterns_dict = {}
        
        # 扫描所有规则
        for framework, regex_patterns in self.compiled_rules.items():
            matched = []
            for pattern in regex_patterns:
                if pattern.search(code):
                    matched.append(pattern.pattern)
            
            if matched:
                # 置信度 = 匹配的模式数 / 总模式数
                confidence = len(matched) / len(regex_patterns)
                matches[framework] = confidence
                matched_patterns_dict[framework] = matched
        
        if not matches:
            return None
        
        # 返回置信度最高的框架
        best_framework = max(matches, key=matches.get)
        confidence = matches[best_framework]
        matched_patterns = matched_patterns_dict[best_framework]
        
        return best_framework, confidence, matched_patterns
    
    def explain(self, code: str) -> Dict:
        """返回详细的检测结果"""
        result = self.detect(code)
        if not result:
            return {"detected": False, "reason": "No rules matched"}
        
        framework, confidence, patterns = result
        return {
            "detected": True,
            "framework": framework,
            "confidence": confidence,
            "method": "rule_based",
            "matched_patterns": patterns,
            "pattern_count": len(patterns),
        }


class AIBasedDetector:
    """基于AI的框架检测（语义理解）"""
    
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        self.frameworks = [
            "react", "vue", "angular", "svelte", "solid", "preact",
            "alpine", "lit", "stencil", "ember", "backbone", "knockout",
            "express", "koa", "fastify", "hapi", "nest", "nextjs",
            "nuxt", "gatsby", "remix", "lodash", "ramda", "underscore"
        ]
    
    def detect(self, code: str) -> Tuple[str, float]:
        """
        基于AI模型检测框架
        
        Returns:
            (framework, confidence)
        """
        # 字符级编码
        encoded = np.array([[ord(c) for c in code[:512]]], dtype=np.int64)
        # 填充到512长度
        if encoded.shape[1] < 512:
            encoded = np.pad(encoded, ((0, 0), (0, 512 - encoded.shape[1])))
        else:
            encoded = encoded[:, :512]
        
        # 推理
        logits = self.session.run(None, {'input_ids': encoded})[0]
        probs = self._softmax(logits[0])
        
        predicted_idx = np.argmax(probs)
        confidence = float(probs[predicted_idx])
        framework = self.frameworks[predicted_idx]
        
        return framework, confidence
    
    def detect_with_all_scores(self, code: str) -> Dict[str, float]:
        """返回所有框架的置信度"""
        encoded = np.array([[ord(c) for c in code[:512]]], dtype=np.int64)
        if encoded.shape[1] < 512:
            encoded = np.pad(encoded, ((0, 0), (0, 512 - encoded.shape[1])))
        else:
            encoded = encoded[:, :512]
        
        logits = self.session.run(None, {'input_ids': encoded})[0]
        probs = self._softmax(logits[0])
        
        return {
            framework: float(prob)
            for framework, prob in zip(self.frameworks, probs)
        }
    
    @staticmethod
    def _softmax(x):
        """Softmax归一化"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


class HybridFrameworkDetector:
    """混合框架检测系统"""
    
    def __init__(self, model_path: str = "models/local/fast_enhanced_quantized.onnx"):
        self.rule_detector = RuleBasedDetector()
        try:
            self.ai_detector = AIBasedDetector(model_path)
            self.ai_available = True
        except Exception as e:
            print(f"⚠️  AI模型加载失败: {e}")
            self.ai_detector = None
            self.ai_available = False
    
    def detect(
        self,
        code: str,
        strategy: str = "hybrid",
        confidence_threshold: float = 0.5
    ) -> DetectionResult:
        """
        检测框架
        
        参数:
            code: 源代码
            strategy: 检测策略 ("rule_only", "ai_only", "hybrid")
            confidence_threshold: 置信度阈值 (低于此值时使用AI兜底)
        
        返回:
            DetectionResult对象
        """
        
        if strategy == "rule_only":
            return self._rule_detect(code)
        
        elif strategy == "ai_only":
            if not self.ai_available:
                raise RuntimeError("AI模型不可用")
            return self._ai_detect(code)
        
        elif strategy == "hybrid":
            return self._hybrid_detect(code, confidence_threshold)
        
        else:
            raise ValueError(f"未知策略: {strategy}")
    
    def _rule_detect(self, code: str) -> DetectionResult:
        """纯规则检测"""
        result = self.rule_detector.detect(code)
        
        if result is None:
            return DetectionResult(
                framework="unknown",
                confidence=0.0,
                method=DetectionMethod.RULE_BASED,
                details={"reason": "No rules matched"}
            )
        
        framework, confidence, patterns = result
        return DetectionResult(
            framework=framework,
            confidence=confidence,
            method=DetectionMethod.RULE_BASED,
            details={
                "matched_patterns": patterns,
                "pattern_count": len(patterns),
            }
        )
    
    def _ai_detect(self, code: str) -> DetectionResult:
        """纯AI检测"""
        framework, confidence = self.ai_detector.detect(code)
        return DetectionResult(
            framework=framework,
            confidence=confidence,
            method=DetectionMethod.AI_BASED,
            details=None
        )
    
    def _hybrid_detect(self, code: str, threshold: float) -> DetectionResult:
        """混合检测: 规则优先，AI兜底"""
        
        # 第一步: 规则检测
        rule_result = self.rule_detector.detect(code)
        
        if rule_result is not None:
            framework, confidence, patterns = rule_result
            
            # 置信度高 → 直接返回
            if confidence >= threshold:
                return DetectionResult(
                    framework=framework,
                    confidence=confidence,
                    method=DetectionMethod.HYBRID,
                    details={
                        "phase": "rule_based_high_confidence",
                        "matched_patterns": patterns,
                    }
                )
            
            # 置信度中等 → 用AI补充
            if self.ai_available:
                ai_framework, ai_confidence = self.ai_detector.detect(code)
                
                # 如果AI和规则一致，返回混合结果
                if ai_framework == framework:
                    combined_confidence = (confidence + ai_confidence) / 2
                    return DetectionResult(
                        framework=framework,
                        confidence=combined_confidence,
                        method=DetectionMethod.HYBRID,
                        details={
                            "phase": "rule_ai_agreement",
                            "rule_confidence": confidence,
                            "ai_confidence": ai_confidence,
                        }
                    )
                
                # 如果不一致，选择置信度更高的
                if ai_confidence > confidence:
                    return DetectionResult(
                        framework=ai_framework,
                        confidence=ai_confidence,
                        method=DetectionMethod.HYBRID,
                        details={
                            "phase": "rule_ai_disagreement_ai_wins",
                            "rule": (framework, confidence),
                            "ai": (ai_framework, ai_confidence),
                        }
                    )
                else:
                    return DetectionResult(
                        framework=framework,
                        confidence=confidence,
                        method=DetectionMethod.HYBRID,
                        details={
                            "phase": "rule_ai_disagreement_rule_wins",
                            "rule": (framework, confidence),
                            "ai": (ai_framework, ai_confidence),
                        }
                    )
            else:
                # AI不可用，返回规则结果
                return DetectionResult(
                    framework=framework,
                    confidence=confidence,
                    method=DetectionMethod.HYBRID,
                    details={
                        "phase": "rule_based_low_confidence_no_ai",
                        "matched_patterns": patterns,
                    }
                )
        
        # 第二步: 规则未匹配，尝试AI
        if self.ai_available:
            framework, confidence = self.ai_detector.detect(code)
            return DetectionResult(
                framework=framework,
                confidence=confidence,
                method=DetectionMethod.HYBRID,
                details={"phase": "ai_based_fallback"}
            )
        
        # 都失败
        return DetectionResult(
            framework="unknown",
            confidence=0.0,
            method=DetectionMethod.HYBRID,
            details={"phase": "all_methods_failed"}
        )
    
    def analyze(self, code: str) -> Dict:
        """详细分析结果"""
        rule_result = self.rule_detector.detect(code)
        
        analysis = {
            "code_length": len(code),
            "rule_based": None,
            "ai_based": None,
            "hybrid": None,
        }
        
        if rule_result:
            framework, confidence, patterns = rule_result
            analysis["rule_based"] = {
                "framework": framework,
                "confidence": confidence,
                "matched_patterns": patterns,
            }
        
        if self.ai_available:
            framework, confidence = self.ai_detector.detect(code)
            analysis["ai_based"] = {
                "framework": framework,
                "confidence": confidence,
            }
        
        hybrid = self.detect(code, strategy="hybrid")
        analysis["hybrid"] = {
            "framework": hybrid.framework,
            "confidence": hybrid.confidence,
            "method": hybrid.method.value,
            "details": hybrid.details,
        }
        
        return analysis


# 使用示例
if __name__ == "__main__":
    detector = HybridFrameworkDetector()
    
    test_cases = [
        ("import React, { useState } from 'react';", "react"),
        ("const Vue = require('vue');", "vue"),
        ("import { Component } from '@angular/core';", "angular"),
        ("const express = require('express');", "express"),
        ("const _ = require('lodash');", "lodash"),
        ("const R = require('ramda');", "ramda"),
    ]
    
    print("\n" + "="*70)
    print("🔀 混合框架检测系统 - 演示")
    print("="*70 + "\n")
    
    for code, expected in test_cases:
        print(f"📝 代码: {code[:50]}...")
        print(f"   期望: {expected}")
        
        result = detector.detect(code, strategy="hybrid")
        print(f"   预测: {result.framework}")
        print(f"   置信度: {result.confidence:.2%}")
        print(f"   方法: {result.method.value}")
        
        if result.details:
            phase = result.details.get("phase", "unknown")
            print(f"   阶段: {phase}")
        
        status = "✅" if result.framework == expected else "❌"
        print(f"   {status}\n")
    
    print("="*70)
    print("💡 混合检测策略:")
    print("   1. 规则检测 (快速、准确)")
    print("   2. 高置信度 → 直接返回")
    print("   3. 低置信度 → AI兜底")
    print("   4. 都失败 → 返回unknown")
    print("="*70 + "\n")
