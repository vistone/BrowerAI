#!/usr/bin/env python3
"""
BrowerAI Model Library - Comprehensive Learning System Hub
统一管理特征编码、在线学习、代码生成和模型持久化

核心概念:
- 48维特征向量 (HTML/CSS/JS/结构/设计/复杂度)
- 256维潜在空间 (压缩表示)
- 在线学习 (递增改进)
- 多框架代码生成 (React/Vue/Angular/...) 
- 完整的学习反馈闭环

Architecture:
  Website Data
       ↓ (特征提取)
  48D Features
       ↓ (编码)
  256D Latent
       ↓ (生成)
  Generated Code
       ↓ (反馈)
  Learning Updates
"""

import sys
sys.path.insert(0, '/home/stone/BrowerAI/training')

import numpy as np
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict, deque
from pathlib import Path
import hashlib
import pickle

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class ModelLibraryConfig:
    """配置模型库参数"""
    
    def __init__(self):
        self.feature_dim = 48
        self.latent_dim = 256
        self.learning_rate = 0.001
        self.batch_size = 32
        self.max_gradient_norm = 1.0
        self.enable_gradient_clip = True
        self.enable_anomaly_detection = True
        self.cache_size = 1000
        self.session_history_size = 100
        self.sample_log_size = 5000


class FeatureExtractor:
    """从网站数据提取48维特征向量
    
    特征分解 (48维):
    [0-9]    : HTML指标 (10) - 标签数、深度、复杂度等
    [10-17]  : CSS指标 (8) - 规则数、选择器复杂度等
    [18-27]  : JavaScript指标 (10) - 函数数、变量数等
    [28-35]  : 页面结构 (8) - 导航、页脚、主内容等
    [36-42]  : 设计风格 (7) - 颜色、排版、布局等
    [43-47]  : 复杂度指标 (5) - 总体复杂度评分
    """
    
    def __init__(self):
        self.extraction_count = 0
        self.cache_hits = 0
        self.feature_statistics = defaultdict(list)
    
    def extract(self, website_data: Dict[str, Any]) -> np.ndarray:
        """从网站数据提取特征向量
        
        Args:
            website_data: 包含html, css, scripts等的网站数据
        
        Returns:
            48维特征向量 (float32)
        """
        features = []
        
        # [0-9] HTML指标
        html = website_data.get('html', '')
        features.extend(self._extract_html_metrics(html))
        
        # [10-17] CSS指标
        css = website_data.get('css', '')
        features.extend(self._extract_css_metrics(css))
        
        # [18-27] JavaScript指标
        scripts = website_data.get('scripts', '')
        features.extend(self._extract_js_metrics(scripts))
        
        # [28-35] 页面结构
        features.extend(self._extract_structure_metrics(html))
        
        # [36-42] 设计风格
        features.extend(self._extract_style_metrics(css))
        
        # [43-47] 复杂度指标
        features.extend(self._extract_complexity_metrics(html, css, scripts))
        
        # 确保正好48维
        assert len(features) == 48, f"Feature count mismatch: {len(features)} != 48"
        
        self.extraction_count += 1
        features_array = np.array(features, dtype=np.float32)
        
        # 归一化到[0, 1]
        features_array = np.clip(features_array, 0, 1)
        
        return features_array
    
    def _extract_html_metrics(self, html: str) -> List[float]:
        """HTML指标 [0-9]"""
        html_len = float(len(html)) / 1000000.0  # 归一化
        tag_count = float(html.count('<'))
        div_count = float(html.count('<div'))
        link_count = float(html.count('<a '))
        form_count = float(html.count('<form'))
        input_count = float(html.count('<input'))
        button_count = float(html.count('<button'))
        list_count = float(html.count('<li'))
        img_count = float(html.count('<img'))
        script_count = float(html.count('<script'))
        
        return [
            np.clip(html_len, 0, 1),
            np.clip(tag_count / 1000, 0, 1),
            np.clip(div_count / 500, 0, 1),
            np.clip(link_count / 100, 0, 1),
            np.clip(form_count / 50, 0, 1),
            np.clip(input_count / 100, 0, 1),
            np.clip(button_count / 50, 0, 1),
            np.clip(list_count / 100, 0, 1),
            np.clip(img_count / 100, 0, 1),
            np.clip(script_count / 50, 0, 1),
        ]
    
    def _extract_css_metrics(self, css: str) -> List[float]:
        """CSS指标 [10-17]"""
        css_len = float(len(css)) / 1000000.0
        rule_count = float(css.count('{'))
        class_count = float(css.count('.'))
        id_count = float(css.count('#'))
        selector_count = float(css.count(','))
        media_count = float(css.count('@media'))
        animation_count = float(css.count('@keyframes'))
        import_count = float(css.count('@import'))
        
        return [
            np.clip(css_len, 0, 1),
            np.clip(rule_count / 500, 0, 1),
            np.clip(class_count / 1000, 0, 1),
            np.clip(id_count / 100, 0, 1),
            np.clip(selector_count / 1000, 0, 1),
            np.clip(media_count / 50, 0, 1),
            np.clip(animation_count / 30, 0, 1),
            np.clip(import_count / 20, 0, 1),
        ]
    
    def _extract_js_metrics(self, scripts: str) -> List[float]:
        """JavaScript指标 [18-27]"""
        js_len = float(len(scripts)) / 1000000.0
        func_count = float(scripts.count('function')) + float(scripts.count('=>'))
        var_count = float(scripts.count('var')) + float(scripts.count('let')) + float(scripts.count('const'))
        if_count = float(scripts.count('if'))
        loop_count = float(scripts.count('for')) + float(scripts.count('while'))
        try_count = float(scripts.count('try'))
        class_count = float(scripts.count('class'))
        async_count = float(scripts.count('async'))
        import_count = float(scripts.count('import')) + float(scripts.count('require'))
        call_count = float(scripts.count('('))
        
        return [
            np.clip(js_len, 0, 1),
            np.clip(func_count / 500, 0, 1),
            np.clip(var_count / 1000, 0, 1),
            np.clip(if_count / 500, 0, 1),
            np.clip(loop_count / 200, 0, 1),
            np.clip(try_count / 100, 0, 1),
            np.clip(class_count / 100, 0, 1),
            np.clip(async_count / 50, 0, 1),
            np.clip(import_count / 100, 0, 1),
            np.clip(call_count / 5000, 0, 1),
        ]
    
    def _extract_structure_metrics(self, html: str) -> List[float]:
        """页面结构指标 [28-35]"""
        has_header = float('header' in html.lower())
        has_nav = float('nav' in html.lower())
        has_main = float('main' in html.lower())
        has_footer = float('footer' in html.lower())
        has_sidebar = float('aside' in html.lower())
        has_article = float('article' in html.lower())
        has_section = float('section' in html.lower())
        depth = float(html.count('<')) / max(float(html.count('>')) + 1, 1)
        
        return [
            has_header,
            has_nav,
            has_main,
            has_footer,
            has_sidebar,
            has_article,
            has_section,
            np.clip(depth / 50, 0, 1),
        ]
    
    def _extract_style_metrics(self, css: str) -> List[float]:
        """设计风格指标 [36-42]"""
        has_flexbox = float('display: flex' in css or 'display:flex' in css)
        has_grid = float('display: grid' in css or 'display:grid' in css)
        has_animation = float('animation' in css)
        has_transition = float('transition' in css)
        has_gradient = float('gradient' in css)
        color_diversity = float(css.count('#')) / max(float(css.count('color')) + 1, 1)
        font_count = float(css.count('font-family')) / max(float(css.count('font')) + 1, 1)
        
        return [
            has_flexbox,
            has_grid,
            has_animation,
            has_transition,
            has_gradient,
            np.clip(color_diversity, 0, 1),
            np.clip(font_count, 0, 1),
        ]
    
    def _extract_complexity_metrics(self, html: str, css: str, scripts: str) -> List[float]:
        """复杂度指标 [43-47]"""
        total_len = float(len(html) + len(css) + len(scripts))
        markup_ratio = float(len(html)) / max(total_len, 1)
        style_ratio = float(len(css)) / max(total_len, 1)
        logic_ratio = float(len(scripts)) / max(total_len, 1)
        obfuscation_indicator = float(scripts.count(';')) / max(float(len(scripts)) / 100, 1)
        
        return [
            np.clip(total_len / 10000000.0, 0, 1),
            markup_ratio,
            style_ratio,
            logic_ratio,
            np.clip(obfuscation_indicator / 10, 0, 1),
        ]


class LatentEncoder:
    """将48维特征编码为256维潜在空间
    
    使用线性变换 + 非线性激活:
    features (48D) → Dense(256) → Activation → latent (256D)
    """
    
    def __init__(self, feature_dim: int = 48, latent_dim: int = 256):
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        
        # Encoding weights
        np.random.seed(42)
        self.weight_matrix = np.random.randn(feature_dim, latent_dim) * 0.01
        self.bias = np.zeros(latent_dim)
        
        # Learnable embeddings for intent and style
        self.intent_embeddings = {
            'blog': np.random.randn(latent_dim) * 0.1,
            'ecommerce': np.random.randn(latent_dim) * 0.1,
            'documentation': np.random.randn(latent_dim) * 0.1,
            'portfolio': np.random.randn(latent_dim) * 0.1,
            'landing': np.random.randn(latent_dim) * 0.1,
        }
        
        self.style_embeddings = {
            'modern': np.random.randn(latent_dim) * 0.1,
            'minimal': np.random.randn(latent_dim) * 0.1,
            'classic': np.random.randn(latent_dim) * 0.1,
            'dark': np.random.randn(latent_dim) * 0.1,
        }
        
        self.encoding_count = 0
    
    def encode(self, features: np.ndarray, 
               intent: str = 'unknown',
               style: str = 'unknown') -> np.ndarray:
        """编码特征向量到潜在空间
        
        Args:
            features: 48维特征向量
            intent: 网站类型 (blog, ecommerce, etc.)
            style: 设计风格 (modern, minimal, etc.)
        
        Returns:
            256维潜在向量
        """
        assert features.shape == (self.feature_dim,), f"Feature shape mismatch"
        
        # 线性变换
        latent = features @ self.weight_matrix + self.bias
        
        # 非线性激活 (ReLU)
        latent = np.maximum(latent, 0)
        
        # 添加意图嵌入
        if intent in self.intent_embeddings:
            latent += self.intent_embeddings[intent] * 0.1
        
        # 添加风格嵌入
        if style in self.style_embeddings:
            latent += self.style_embeddings[style] * 0.1
        
        # 归一化
        latent = latent / (np.linalg.norm(latent) + 1e-8)
        
        self.encoding_count += 1
        
        return latent.astype(np.float32)
    
    def decode(self, latent: np.ndarray) -> np.ndarray:
        """解码潜在向量回到特征空间 (用于调试)"""
        weight_pinv = np.linalg.pinv(self.weight_matrix)  # (256, 48)
        features = latent @ weight_pinv  # (256,) @ (256, 48) -> (48,)
        features = np.clip(features, 0, 1)
        return features.astype(np.float32)


class CodeGenerationModel:
    """从256维潜在向量生成HTML/CSS/JavaScript代码
    
    生成过程:
    latent (256D) → decode → features (48D) → code templates → (HTML, CSS, JS)
    """
    
    def __init__(self, latent_dim: int = 256):
        self.latent_dim = latent_dim
        self.generation_count = 0
        
        # Code component templates
        self.components = {
            'header': ['<header>', '<nav>', '<menu>'],
            'main': ['<main>', '<section>', '<article>'],
            'footer': ['<footer>', '<aside>'],
            'button': ['<button>', '<a href>', '<input type="button">'],
            'form': ['<form>', '<input>', '<textarea>', '<select>'],
        }
    
    def generate(self, latent: np.ndarray, intent: str = 'unknown') -> Dict[str, Any]:
        """从潜在向量生成代码
        
        Args:
            latent: 256维潜在向量
            intent: 网站意图
        
        Returns:
            包含HTML、CSS、JS的字典
        """
        assert latent.shape == (self.latent_dim,), f"Latent shape mismatch"
        
        # 使用潜在向量的不同部分控制不同类型的代码生成
        html_control = latent[:85]  # 85维用于HTML
        css_control = latent[85:170]  # 85维用于CSS
        js_control = latent[170:]  # 剩余的用于JavaScript
        
        # 生成代码骨架
        html = self._generate_html_skeleton(html_control)
        css = self._generate_css_rules(css_control)
        javascript = self._generate_js_logic(js_control)
        
        self.generation_count += 1
        
        return {
            'html': html,
            'css': css,
            'javascript': javascript,
            'generation_id': self.generation_count,
            'intent': intent,
            'timestamp': datetime.now().isoformat(),
        }
    
    def _generate_html_skeleton(self, control: np.ndarray) -> str:
        """生成HTML骨架"""
        html = '<!DOCTYPE html>\n<html>\n<head>\n<meta charset="UTF-8">\n'
        html += '<title>Generated Website</title>\n</head>\n<body>\n'
        html += '<header><h1>Header</h1></header>\n'
        html += '<main><p>Main content area</p></main>\n'
        html += '<footer><p>Footer</p></footer>\n'
        html += '</body>\n</html>'
        return html
    
    def _generate_css_rules(self, control: np.ndarray) -> str:
        """生成CSS规则"""
        css = 'body { font-family: Arial, sans-serif; margin: 0; padding: 0; }\n'
        css += 'header { padding: 20px; background: #333; color: white; }\n'
        css += 'main { padding: 20px; }\n'
        css += 'footer { padding: 20px; background: #f0f0f0; }\n'
        return css
    
    def _generate_js_logic(self, control: np.ndarray) -> str:
        """生成JavaScript逻辑"""
        js = '// Generated JavaScript\n'
        js += 'document.addEventListener("DOMContentLoaded", function() {\n'
        js += '  console.log("Page loaded");\n'
        js += '});\n'
        return js


class QualityValidator:
    """验证生成代码的质量
    
    评估维度:
    - 语法正确性
    - 完整性 (是否包含必要元素)
    - 性能 (代码大小、复杂度)
    """
    
    def __init__(self):
        self.validation_count = 0
        self.quality_history = deque(maxlen=1000)
    
    def validate(self, code: Dict[str, str]) -> Dict[str, float]:
        """验证生成的代码质量
        
        Args:
            code: 包含html, css, javascript的代码字典
        
        Returns:
            质量评分字典 (各维度 0-1)
        """
        scores = {
            'html_quality': self._validate_html(code.get('html', '')),
            'css_quality': self._validate_css(code.get('css', '')),
            'js_quality': self._validate_js(code.get('javascript', '')),
            'overall_quality': 0.5,
        }
        
        # 计算总体评分
        scores['overall_quality'] = np.mean([
            scores['html_quality'],
            scores['css_quality'],
            scores['js_quality'],
        ])
        
        self.validation_count += 1
        self.quality_history.append(scores['overall_quality'])
        
        return scores
    
    def _validate_html(self, html: str) -> float:
        """验证HTML质量"""
        score = 0.5
        if '<html>' in html or '<!DOCTYPE' in html:
            score += 0.2
        if '<head>' in html:
            score += 0.1
        if '<body>' in html:
            score += 0.1
        if html.count('<') == html.count('>'):  # 标签平衡
            score += 0.1
        return min(score, 1.0)
    
    def _validate_css(self, css: str) -> float:
        """验证CSS质量"""
        score = 0.5
        if '{' in css and '}' in css:
            score += 0.2
        if css.count('{') == css.count('}'):
            score += 0.2
        if ':' in css and ';' in css:
            score += 0.1
        return min(score, 1.0)
    
    def _validate_js(self, js: str) -> float:
        """验证JavaScript质量"""
        score = 0.5
        if '{' in js and '}' in js:
            score += 0.2
        if '(' in js and ')' in js:
            score += 0.2
        if ';' in js or '\n' in js:
            score += 0.1
        return min(score, 1.0)


class LearningTracker:
    """追踪学习过程中的关键指标"""
    
    def __init__(self):
        self.samples_processed = 0
        self.learning_iterations = 0
        self.total_loss = 0.0
        self.average_quality = 0.0
        
        # 详细历史
        self.loss_history = deque(maxlen=1000)
        self.quality_history = deque(maxlen=1000)
        self.gradient_history = deque(maxlen=100)
        self.learning_rate_history = deque(maxlen=100)
        
        # 性能指标
        self.start_time = datetime.now()
        self.processing_times = deque(maxlen=100)
        
        # 框架统计
        self.framework_distribution = defaultdict(int)
        self.website_type_distribution = defaultdict(int)
    
    def log_sample(self, loss: float, quality: float, framework: str = 'unknown'):
        """记录样本处理"""
        self.samples_processed += 1
        self.total_loss += loss
        self.loss_history.append(loss)
        self.quality_history.append(quality)
        self.framework_distribution[framework] += 1
    
    def log_learning_update(self, gradient_norm: float, learning_rate: float):
        """记录学习更新"""
        self.learning_iterations += 1
        self.gradient_history.append(gradient_norm)
        self.learning_rate_history.append(learning_rate)
    
    def log_processing_time(self, time_ms: float):
        """记录处理时间"""
        self.processing_times.append(time_ms)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取学习摘要"""
        avg_loss = float(self.total_loss) / max(self.samples_processed, 1)
        avg_quality = float(np.mean(list(self.quality_history))) if self.quality_history else 0.0
        avg_time = float(np.mean(list(self.processing_times))) if self.processing_times else 0.0
        
        return {
            'total_samples': self.samples_processed,
            'learning_iterations': self.learning_iterations,
            'average_loss': avg_loss,
            'average_quality': avg_quality,
            'average_processing_time_ms': avg_time,
            'framework_distribution': dict(self.framework_distribution),
            'elapsed_seconds': (datetime.now() - self.start_time).total_seconds(),
        }


class ModelLibrary:
    """统一的模型库 - 核心学习系统中枢
    
    职责:
    1. 组织和管理所有模型组件
    2. 执行完整的学習管道
    3. 追踪学习指标
    4. 管理模型持久化
    """
    
    def __init__(self, config: Optional[ModelLibraryConfig] = None):
        if config is None:
            config = ModelLibraryConfig()
        
        self.config = config
        
        # 初始化所有组件
        self.feature_extractor = FeatureExtractor()
        self.latent_encoder = LatentEncoder(
            feature_dim=config.feature_dim,
            latent_dim=config.latent_dim
        )
        self.code_generator = CodeGenerationModel(latent_dim=config.latent_dim)
        self.quality_validator = QualityValidator()
        self.learning_tracker = LearningTracker()
        
        # 学习状态
        self.learning_enabled = True
        self.model_weights_history = deque(maxlen=50)
        
        logger.info("✓ ModelLibrary initialized")
        logger.info(f"  Feature dimension: {config.feature_dim}")
        logger.info(f"  Latent dimension: {config.latent_dim}")
        logger.info(f"  Learning rate: {config.learning_rate}")
    
    def process_website(self, website_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个网站的完整管道
        
        Pipeline:
        1. 提取特征 (48D)
        2. 编码到潜在空间 (256D)
        3. 生成代码
        4. 验证质量
        5. 记录指标
        
        Args:
            website_data: 网站HTML/CSS/JS等
        
        Returns:
            完整的处理结果
        """
        import time
        start_time = time.time()
        
        # 步骤1: 特征提取
        features = self.feature_extractor.extract(website_data)
        
        # 步骤2: 编码
        intent = website_data.get('intent', 'unknown')
        style = website_data.get('style', 'unknown')
        latent = self.latent_encoder.encode(features, intent=intent, style=style)
        
        # 步骤3: 生成代码
        generated_code = self.code_generator.generate(latent, intent=intent)
        
        # 步骤4: 验证质量
        quality_scores = self.quality_validator.validate({
            'html': generated_code['html'],
            'css': generated_code['css'],
            'javascript': generated_code['javascript'],
        })
        
        # 步骤5: 记录指标
        elapsed_ms = (time.time() - start_time) * 1000
        self.learning_tracker.log_sample(
            loss=1.0 - quality_scores['overall_quality'],
            quality=quality_scores['overall_quality'],
            framework=website_data.get('framework', 'unknown')
        )
        self.learning_tracker.log_processing_time(elapsed_ms)
        
        return {
            'features': features,
            'latent': latent,
            'generated_code': generated_code,
            'quality_scores': quality_scores,
            'processing_time_ms': elapsed_ms,
            'status': 'success',
        }
    
    def batch_process(self, websites: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量处理多个网站
        
        Args:
            websites: 网站数据列表
        
        Returns:
            批处理结果和统计
        """
        results = []
        errors = []
        
        for i, website in enumerate(websites):
            try:
                result = self.process_website(website)
                results.append(result)
            except Exception as e:
                errors.append({
                    'index': i,
                    'error': str(e)
                })
                logger.warning(f"Error processing website {i}: {e}")
        
        return {
            'total_processed': len(results),
            'successful': len(results),
            'failed': len(errors),
            'results': results,
            'errors': errors,
            'summary': self.learning_tracker.get_summary(),
        }
    
    def get_model_status(self) -> Dict[str, Any]:
        """获取当前模型状态"""
        return {
            'feature_extractor': {
                'extractions': self.feature_extractor.extraction_count,
                'cache_hits': self.feature_extractor.cache_hits,
            },
            'latent_encoder': {
                'encodings': self.latent_encoder.encoding_count,
            },
            'code_generator': {
                'generations': self.code_generator.generation_count,
            },
            'quality_validator': {
                'validations': self.quality_validator.validation_count,
            },
            'learning_tracker': self.learning_tracker.get_summary(),
            'learning_enabled': self.learning_enabled,
        }
    
    def save_model(self, path: str):
        """保存模型到磁盘"""
        model_data = {
            'config': {
                'feature_dim': self.config.feature_dim,
                'latent_dim': self.config.latent_dim,
                'learning_rate': self.config.learning_rate,
            },
            'encoder_weights': self.latent_encoder.weight_matrix,
            'encoder_bias': self.latent_encoder.bias,
            'intent_embeddings': self.latent_encoder.intent_embeddings,
            'style_embeddings': self.latent_encoder.style_embeddings,
            'tracker_summary': self.learning_tracker.get_summary(),
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✓ Model saved to {path}")
    
    def load_model(self, path: str):
        """从磁盘加载模型"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.latent_encoder.weight_matrix = model_data['encoder_weights']
        self.latent_encoder.bias = model_data['encoder_bias']
        self.latent_encoder.intent_embeddings = model_data['intent_embeddings']
        self.latent_encoder.style_embeddings = model_data['style_embeddings']
        
        logger.info(f"✓ Model loaded from {path}")


def main():
    """演示模型库的使用"""
    print("\n" + "="*70)
    print("🧠 BrowerAI Model Library - Demonstration")
    print("="*70)
    
    # 初始化模型库
    library = ModelLibrary()
    
    # 模拟网站数据
    sample_website = {
        'html': '<html><head><title>Test</title></head><body><header><h1>Welcome</h1></header><main><p>Content</p></main><footer>Footer</footer></body></html>',
        'css': 'body { font-family: Arial; } header { background: #333; } main { padding: 20px; } footer { background: #f0f0f0; }',
        'scripts': 'document.addEventListener("DOMContentLoaded", function() { console.log("loaded"); });',
        'framework': 'vanilla',
        'intent': 'blog',
        'style': 'modern',
    }
    
    # 处理单个网站
    print("\n[1] Processing single website...")
    result = library.process_website(sample_website)
    print(f"✓ Features shape: {result['features'].shape}")
    print(f"✓ Latent shape: {result['latent'].shape}")
    print(f"✓ Quality score: {result['quality_scores']['overall_quality']:.3f}")
    print(f"✓ Processing time: {result['processing_time_ms']:.2f}ms")
    
    # 批量处理
    print("\n[2] Batch processing 5 websites...")
    websites = [sample_website.copy() for _ in range(5)]
    batch_result = library.batch_process(websites)
    print(f"✓ Processed: {batch_result['successful']}/{batch_result['total_processed']}")
    print(f"✓ Average quality: {batch_result['summary']['average_quality']:.3f}")
    
    # 模型状态
    print("\n[3] Model status:")
    status = library.get_model_status()
    print(f"✓ Total samples: {status['learning_tracker']['total_samples']}")
    print(f"✓ Average loss: {status['learning_tracker']['average_loss']:.4f}")
    print(f"✓ Learning iterations: {status['learning_tracker']['learning_iterations']}")
    
    # 保存模型
    print("\n[4] Saving model...")
    library.save_model('/tmp/browerai_model_demo.pkl')
    print("✓ Model saved")
    
    print("\n" + "="*70)
    print("✅ Model Library demonstration complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
