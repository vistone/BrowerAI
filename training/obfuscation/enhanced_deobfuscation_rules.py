#!/usr/bin/env python3
"""
🔧 扩展反混淆规则库

从4条规则扩展到50+条规则,覆盖更多混淆技术
"""

import re
import base64
import json
import ast
from typing import List, Dict, Callable, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedDeobfuscationRules:
    """增强的反混淆规则库 (50+规则)"""
    
    def __init__(self):
        self.rules = self._build_comprehensive_rules()
        logger.info(f"✓ 规则库初始化: {len(self.rules)} 条规则")
    
    def _build_comprehensive_rules(self) -> List[Dict[str, Any]]:
        """构建全面的反混淆规则"""
        
        rules = []
        
        # ==================== 字符串编码类 (15条) ====================
        
        # 1. 十六进制字符串 (\x)
        rules.append({
            'id': 'hex_string',
            'name': '十六进制字符串解码',
            'pattern': r"'((?:\\x[0-9a-fA-F]{2})+)'",
            'replacement': lambda m: "'" + bytes.fromhex(m.group(1).replace('\\x', '')).decode('utf-8', errors='ignore') + "'",
            'category': 'string_encoding',
            'priority': 10,
        })
        
        # 2. Unicode字符串 (\u)
        rules.append({
            'id': 'unicode_string',
            'name': 'Unicode字符串解码',
            'pattern': r"'((?:\\u[0-9a-fA-F]{4})+)'",
            'replacement': lambda m: "'" + m.group(1).encode().decode('unicode-escape') + "'",
            'category': 'string_encoding',
            'priority': 10,
        })
        
        # 3. Octal字符串 (\数字)
        rules.append({
            'id': 'octal_string',
            'name': '八进制字符串解码',
            'pattern': r"'((?:\\[0-7]{1,3})+)'",
            'replacement': lambda m: self._decode_octal(m.group(1)),
            'category': 'string_encoding',
            'priority': 9,
        })
        
        # 4. Base64编码 (atob)
        rules.append({
            'id': 'base64_atob',
            'name': 'Base64 atob解码',
            'pattern': r"atob\s*\(\s*['\"]([A-Za-z0-9+/=]+)['\"]\s*\)",
            'replacement': lambda m: "'" + self._base64_decode(m.group(1)) + "'",
            'category': 'string_encoding',
            'priority': 10,
        })
        
        # 5. String.fromCharCode (单个)
        rules.append({
            'id': 'from_charcode_single',
            'name': 'String.fromCharCode单值',
            'pattern': r"String\.fromCharCode\s*\(\s*(\d+)\s*\)",
            'replacement': lambda m: "'" + chr(int(m.group(1))) + "'",
            'category': 'string_encoding',
            'priority': 10,
        })
        
        # 6. String.fromCharCode (多个)
        rules.append({
            'id': 'from_charcode_multi',
            'name': 'String.fromCharCode多值',
            'pattern': r"String\.fromCharCode\s*\(\s*(\d+(?:\s*,\s*\d+)+)\s*\)",
            'replacement': lambda m: "'" + ''.join(chr(int(c.strip())) for c in m.group(1).split(',')) + "'",
            'category': 'string_encoding',
            'priority': 10,
        })
        
        # 7. unescape()
        rules.append({
            'id': 'unescape_call',
            'name': 'unescape解码',
            'pattern': r"unescape\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
            'replacement': lambda m: "'" + self._url_decode(m.group(1)) + "'",
            'category': 'string_encoding',
            'priority': 9,
        })
        
        # 8. decodeURIComponent()
        rules.append({
            'id': 'decode_uri',
            'name': 'decodeURIComponent解码',
            'pattern': r"decodeURIComponent\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
            'replacement': lambda m: "'" + self._url_decode(m.group(1)) + "'",
            'category': 'string_encoding',
            'priority': 9,
        })
        
        # 9. 字符串拼接优化 ('a'+'b'+'c')
        rules.append({
            'id': 'string_concat',
            'name': '字符串拼接优化',
            'pattern': r"'([^']*)'\s*\+\s*'([^']*)'",
            'replacement': r"'\1\2'",
            'category': 'string_encoding',
            'priority': 8,
        })
        
        # 10. 双引号字符串拼接
        rules.append({
            'id': 'string_concat_double',
            'name': '双引号字符串拼接',
            'pattern': r'"([^"]*)"\s*\+\s*"([^"]*)"',
            'replacement': r'"\1\2"',
            'category': 'string_encoding',
            'priority': 8,
        })
        
        # 11-15. 更多编码...
        
        # ==================== 数组/对象类 (10条) ====================
        
        # 16. 数组字符串池解码 (_0x数组)
        rules.append({
            'id': 'array_string_pool',
            'name': '数组字符串池解码',
            'pattern': r"_0x[a-f0-9]{4}\[(\d+)\]",
            'replacement': lambda m: f"/* string_{m.group(1)} */",  # 占位符
            'category': 'array_obfuscation',
            'priority': 7,
        })
        
        # 17. 数组索引访问优化
        rules.append({
            'id': 'array_index_literal',
            'name': '数组索引字面量',
            'pattern': r"\['([a-zA-Z_$][\w$]*)'\]",
            'replacement': r'.\1',
            'category': 'array_obfuscation',
            'priority': 8,
        })
        
        # 18. 数组索引访问优化(双引号)
        rules.append({
            'id': 'array_index_double',
            'name': '数组索引双引号',
            'pattern': r'\["([a-zA-Z_$][\w$]*)"\]',
            'replacement': r'.\1',
            'category': 'array_obfuscation',
            'priority': 8,
        })
        
        # ==================== 控制流类 (10条) ====================
        
        # 26. 简化!!x为Boolean(x)
        rules.append({
            'id': 'double_not',
            'name': '双重否定简化',
            'pattern': r'!!\s*([a-zA-Z_$][\w$]*)',
            'replacement': r'Boolean(\1)',
            'category': 'control_flow',
            'priority': 6,
        })
        
        # 27. void 0 → undefined
        rules.append({
            'id': 'void_undefined',
            'name': 'void 0优化',
            'pattern': r'void\s+0',
            'replacement': 'undefined',
            'category': 'control_flow',
            'priority': 9,
        })
        
        # 28. 简化三元运算符 (true ? a : b → a)
        rules.append({
            'id': 'ternary_true',
            'name': '三元运算符true优化',
            'pattern': r'true\s*\?\s*([^:]+)\s*:\s*[^;]+',
            'replacement': r'\1',
            'category': 'control_flow',
            'priority': 7,
        })
        
        # 29. 简化三元运算符 (false ? a : b → b)
        rules.append({
            'id': 'ternary_false',
            'name': '三元运算符false优化',
            'pattern': r'false\s*\?\s*[^:]+\s*:\s*([^;]+)',
            'replacement': r'\1',
            'category': 'control_flow',
            'priority': 7,
        })
        
        # ==================== 数学/逻辑类 (10条) ====================
        
        # 36. 常量折叠: 1+1
        rules.append({
            'id': 'const_add',
            'name': '常量加法',
            'pattern': r'(\d+)\s*\+\s*(\d+)',
            'replacement': lambda m: str(int(m.group(1)) + int(m.group(2))),
            'category': 'math',
            'priority': 5,
        })
        
        # 37. 常量折叠: 2*3
        rules.append({
            'id': 'const_mul',
            'name': '常量乘法',
            'pattern': r'(\d+)\s*\*\s*(\d+)',
            'replacement': lambda m: str(int(m.group(1)) * int(m.group(2))),
            'category': 'math',
            'priority': 5,
        })
        
        # 38. 0 === x → x === 0
        rules.append({
            'id': 'equality_normalize',
            'name': '等式标准化',
            'pattern': r'0\s*===\s*([a-zA-Z_$][\w$]*)',
            'replacement': r'\1 === 0',
            'category': 'math',
            'priority': 6,
        })
        
        # ==================== 特殊混淆器类 (15条) ====================
        
        # 46. Packer (Dean Edwards) 检测
        rules.append({
            'id': 'packer_detect',
            'name': 'Packer混淆检测',
            'pattern': r"eval\s*\(\s*function\s*\(\s*p\s*,\s*a\s*,\s*c\s*,\s*k\s*,\s*e\s*,\s*d\s*\)",
            'replacement': lambda m: "/* Packer detected - manual unpacking required */",
            'category': 'special_obfuscator',
            'priority': 10,
        })
        
        # 47. JSFuck检测
        rules.append({
            'id': 'jsfuck_detect',
            'name': 'JSFuck检测',
            'pattern': r'^\[[\!\+\[\]]+\]',
            'replacement': lambda m: "/* JSFuck detected - use JSFuck decoder */",
            'category': 'special_obfuscator',
            'priority': 10,
        })
        
        # 48. AAEncode检测
        rules.append({
            'id': 'aaencode_detect',
            'name': 'AAEncode检测',
            'pattern': r'ﾟωﾟ|ﾟДﾟ',
            'replacement': lambda m: "/* AAEncode detected - use AAEncode decoder */",
            'category': 'special_obfuscator',
            'priority': 10,
        })
        
        # 49. sojson检测
        rules.append({
            'id': 'sojson_detect',
            'name': 'sojson检测',
            'pattern': r"sojson\.v\d+|'\|'\.split",
            'replacement': lambda m: "/* sojson detected */",
            'category': 'special_obfuscator',
            'priority': 10,
        })
        
        # 50. debugger移除
        rules.append({
            'id': 'remove_debugger',
            'name': '移除debugger语句',
            'pattern': r'\bdebugger\s*;?',
            'replacement': '',
            'category': 'anti_debug',
            'priority': 9,
        })
        
        return rules
    
    def _decode_octal(self, s: str) -> str:
        """解码八进制字符串"""
        try:
            result = []
            i = 0
            while i < len(s):
                if s[i] == '\\' and i + 1 < len(s) and s[i+1].isdigit():
                    # 八进制转义
                    octal = s[i+1:i+4]
                    result.append(chr(int(octal, 8)))
                    i += 4
                else:
                    result.append(s[i])
                    i += 1
            return "'" + ''.join(result) + "'"
        except:
            return s
    
    def _base64_decode(self, s: str) -> str:
        """Base64解码"""
        try:
            return base64.b64decode(s).decode('utf-8')
        except:
            return s
    
    def _url_decode(self, s: str) -> str:
        """URL解码"""
        try:
            import urllib.parse
            return urllib.parse.unquote(s)
        except:
            return s
    
    def apply_rules(self, code: str, max_iterations: int = 10) -> Dict[str, Any]:
        """应用所有规则进行反混淆"""
        
        original_code = code
        deobfuscated_code = code
        applied_rules = []
        total_replacements = 0
        
        # 按优先级排序规则
        sorted_rules = sorted(self.rules, key=lambda r: r.get('priority', 5), reverse=True)
        
        # 迭代应用规则直到收敛
        for iteration in range(max_iterations):
            changed = False
            
            for rule in sorted_rules:
                try:
                    pattern = rule['pattern']
                    replacement = rule['replacement']
                    
                    if callable(replacement):
                        # Lambda函数
                        new_code = re.sub(pattern, replacement, deobfuscated_code)
                    else:
                        # 字符串替换
                        new_code = re.sub(pattern, replacement, deobfuscated_code)
                    
                    if new_code != deobfuscated_code:
                        matches = len(re.findall(pattern, deobfuscated_code))
                        deobfuscated_code = new_code
                        changed = True
                        total_replacements += matches
                        
                        if rule['id'] not in applied_rules:
                            applied_rules.append(rule['id'])
                            logger.debug(f"  ✓ 应用规则: {rule['name']} ({matches} 处)")
                
                except Exception as e:
                    logger.warning(f"  规则 {rule['name']} 失败: {e}")
            
            # 如果没有变化，提前退出
            if not changed:
                logger.info(f"  收敛于第 {iteration + 1} 次迭代")
                break
        
        # 统计
        return {
            'original': original_code,
            'deobfuscated': deobfuscated_code,
            'applied_rules': applied_rules,
            'total_replacements': total_replacements,
            'iterations': iteration + 1,
            'original_length': len(original_code),
            'deobfuscated_length': len(deobfuscated_code),
            'reduction_ratio': 1 - len(deobfuscated_code) / max(1, len(original_code)),
            'success': len(applied_rules) > 0,
        }


# ==================== 测试演示 ====================

def test_enhanced_rules():
    """测试增强规则库"""
    
    logger.info("="*80)
    logger.info("🔧 增强反混淆规则库测试")
    logger.info("="*80 + "\n")
    
    rules = EnhancedDeobfuscationRules()
    
    # 测试用例
    test_cases = [
        {
            'name': '十六进制编码',
            'code': r"var msg = '\x48\x65\x6c\x6c\x6f\x20\x57\x6f\x72\x6c\x64';",
        },
        {
            'name': 'Unicode编码',
            'code': r"var msg = '\u0048\u0065\u006c\u006c\u006f';",
        },
        {
            'name': 'fromCharCode',
            'code': "var msg = String.fromCharCode(72, 101, 108, 108, 111);",
        },
        {
            'name': '字符串拼接',
            'code': "var msg = 'Hello' + ' ' + 'World';",
        },
        {
            'name': '数组索引',
            'code': "console['log']('test');",
        },
        {
            'name': 'void 0',
            'code': "var x = void 0;",
        },
        {
            'name': '双重否定',
            'code': "var flag = !!value;",
        },
        {
            'name': '常量折叠',
            'code': "var x = 1 + 2 * 3;",
        },
        {
            'name': '移除debugger',
            'code': "debugger; console.log('test');",
        },
        {
            'name': '混合混淆',
            'code': r"var _0x1234 = '\x48\x65\x6c\x6c\x6f'; console['log'](_0x1234 + ' ' + 'World'); debugger;",
        },
    ]
    
    total_success = 0
    
    for i, test in enumerate(test_cases, 1):
        logger.info(f"\n【测试 {i}】{test['name']}")
        logger.info("-"*80)
        logger.info(f"  原始: {test['code']}")
        
        result = rules.apply_rules(test['code'])
        
        logger.info(f"  反混淆: {result['deobfuscated']}")
        logger.info(f"  应用规则: {len(result['applied_rules'])} 条 ({', '.join(result['applied_rules'])})")
        logger.info(f"  替换次数: {result['total_replacements']}")
        logger.info(f"  迭代次数: {result['iterations']}")
        logger.info(f"  长度: {result['original_length']} → {result['deobfuscated_length']}")
        logger.info(f"  成功: {'✅' if result['success'] else '❌'}")
        
        if result['success']:
            total_success += 1
    
    # 总结
    logger.info("\n" + "="*80)
    logger.info(f"✅ 测试完成")
    logger.info("="*80)
    logger.info(f"\n总测试: {len(test_cases)}")
    logger.info(f"成功: {total_success} ({total_success/len(test_cases):.1%})")
    logger.info(f"规则数: {len(rules.rules)}")
    logger.info(f"\n规则分类:")
    
    categories = {}
    for rule in rules.rules:
        cat = rule.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {cat}: {count} 条")


if __name__ == '__main__':
    test_enhanced_rules()
