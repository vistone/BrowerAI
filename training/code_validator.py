#!/usr/bin/env python3
"""
Code Generation Validation System - Validates generated HTML/CSS/JS
Ensures code quality, correctness, and constraint satisfaction

Features:
- HTML syntax validation
- CSS validation
- JavaScript validation
- Semantic consistency checks
- Performance analysis
- Security scanning
- Constraint satisfaction
"""

import logging
import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of code validation"""
    is_valid: bool
    code_type: str  # html, css, or javascript
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_valid': self.is_valid,
            'code_type': self.code_type,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'errors': self.errors[:5],  # Top 5 errors
            'warnings': self.warnings[:5],  # Top 5 warnings
            'metrics': self.metrics,
            'score': self.score
        }


class HTMLValidator:
    """Validates generated HTML code"""
    
    def __init__(self):
        self.max_nesting_depth = 20
        self.max_file_size = 500000  # 500KB
        
    def validate(self, html: str) -> ValidationResult:
        """Validate HTML code"""
        result = ValidationResult(is_valid=True, code_type="html")
        
        try:
            # Check size
            if len(html) > self.max_file_size:
                result.errors.append(
                    f"HTML size {len(html)} bytes exceeds limit {self.max_file_size}"
                )
            
            # Check for required tags
            if not html.strip():
                result.errors.append("Empty HTML")
            elif '<html' not in html.lower() and '<body' not in html.lower():
                result.warnings.append("Missing html/body tags")
            
            # Check tag balance
            tag_balance = self._check_tag_balance(html)
            if tag_balance != 0:
                result.errors.append(
                    f"Unbalanced HTML tags (balance: {tag_balance})"
                )
            
            # Check nesting depth
            max_depth = self._check_nesting_depth(html)
            result.metrics['max_nesting_depth'] = max_depth
            if max_depth > self.max_nesting_depth:
                result.warnings.append(
                    f"Excessive nesting depth: {max_depth} (limit: {self.max_nesting_depth})"
                )
            
            # Check for common accessibility issues
            if '<img ' in html and not re.search(r'<img[^>]+alt=', html):
                result.warnings.append("Images missing alt text")
            
            # Check for deprecated tags
            deprecated = self._find_deprecated_tags(html)
            if deprecated:
                result.warnings.extend([
                    f"Deprecated tag found: {tag}" for tag in deprecated
                ])
            
            # Calculate score
            result.score = max(0, 1.0 - len(result.errors) * 0.2 - len(result.warnings) * 0.05)
            result.is_valid = len(result.errors) == 0
            
        except Exception as e:
            logger.error(f"HTML validation error: {e}")
            result.errors.append(str(e))
            result.is_valid = False
        
        return result
    
    def _check_tag_balance(self, html: str) -> int:
        """Check if tags are balanced"""
        # Self-closing tags
        self_closing = {'br', 'hr', 'img', 'input', 'meta', 'link', 'area', 'base', 'col', 'embed', 'source', 'track', 'wbr'}
        
        balance = 0
        pattern = r'</?(\w+)'
        
        for match in re.finditer(pattern, html):
            tag = match.group(1).lower()
            
            if tag in self_closing:
                continue
            
            if match.group(0).startswith('</'):
                balance -= 1
            else:
                balance += 1
        
        return balance
    
    def _check_nesting_depth(self, html: str) -> int:
        """Check maximum nesting depth"""
        max_depth = 0
        current_depth = 0
        
        for match in re.finditer(r'</?(\w+)', html):
            if match.group(0).startswith('</'):
                current_depth -= 1
            else:
                current_depth += 1
                max_depth = max(max_depth, current_depth)
        
        return max_depth
    
    def _find_deprecated_tags(self, html: str) -> List[str]:
        """Find deprecated HTML tags"""
        deprecated_tags = {'center', 'font', 'big', 'small', 'strike', 'u', 'acronym', 'applet', 'blink', 'marquee'}
        found = []
        
        for tag in deprecated_tags:
            if re.search(rf'<{tag}[\s>]', html, re.IGNORECASE):
                found.append(tag)
        
        return found


class CSSValidator:
    """Validates generated CSS code"""
    
    def __init__(self):
        self.max_file_size = 500000  # 500KB
    
    def validate(self, css: str) -> ValidationResult:
        """Validate CSS code"""
        result = ValidationResult(is_valid=True, code_type="css")
        
        try:
            # Check size
            if len(css) > self.max_file_size:
                result.errors.append(
                    f"CSS size {len(css)} bytes exceeds limit {self.max_file_size}"
                )
            
            # Check if empty
            if not css.strip():
                result.warnings.append("Empty CSS")
                return result
            
            # Check for balanced braces
            brace_balance = css.count('{') - css.count('}')
            if brace_balance != 0:
                result.errors.append(f"Unbalanced CSS braces (balance: {brace_balance})")
            
            # Check for common issues
            self._check_color_formats(css, result)
            self._check_selector_complexity(css, result)
            self._check_property_validity(css, result)
            
            # Count rules
            rule_count = len(re.findall(r'\{', css))
            result.metrics['rule_count'] = rule_count
            
            # Calculate score
            result.score = max(0, 1.0 - len(result.errors) * 0.2 - len(result.warnings) * 0.05)
            result.is_valid = len(result.errors) == 0
            
        except Exception as e:
            logger.error(f"CSS validation error: {e}")
            result.errors.append(str(e))
            result.is_valid = False
        
        return result
    
    def _check_color_formats(self, css: str, result: ValidationResult):
        """Check for valid color formats"""
        # Match color-related properties
        color_pattern = r'(?:color|background|border):\s*([^;]+)'
        
        for match in re.finditer(color_pattern, css):
            color_value = match.group(1).strip()
            
            # Basic color format validation
            if not self._is_valid_color(color_value):
                if len(result.warnings) < 10:  # Limit warnings
                    result.warnings.append(f"Suspicious color value: {color_value[:30]}")
    
    def _is_valid_color(self, color: str) -> bool:
        """Check if color is valid"""
        # Check for common color names or formats
        valid_formats = [
            r'^#[0-9A-Fa-f]{3}$',  # #RGB
            r'^#[0-9A-Fa-f]{6}$',  # #RRGGBB
            r'^rgb\(',  # rgb(...)
            r'^rgba\(',  # rgba(...)
            r'^hsl\(',  # hsl(...)
            r'^hsla\(',  # hsla(...)
            r'^transparent$',
            r'^currentColor$',
            r'^inherit$'
        ]
        
        # Check named colors
        named_colors = {'white', 'black', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta'}
        
        color_lower = color.lower().strip()
        
        if color_lower in named_colors:
            return True
        
        for pattern in valid_formats:
            if re.match(pattern, color_lower):
                return True
        
        return False
    
    def _check_selector_complexity(self, css: str, result: ValidationResult):
        """Check selector complexity"""
        # Extract selectors (before {)
        selectors = re.findall(r'^[^{]+(?={)', css, re.MULTILINE)
        
        complex_selectors = 0
        for selector in selectors:
            # Count combinators
            combinator_count = len(re.findall(r'[+~>]', selector))
            if combinator_count > 3:
                complex_selectors += 1
        
        if complex_selectors > 0:
            result.metrics['complex_selectors'] = complex_selectors
    
    def _check_property_validity(self, css: str, result: ValidationResult):
        """Check for invalid CSS properties"""
        # Find common typo patterns
        common_typos = {
            r'coolor:': 'color',
            r'backgrond:': 'background',
            r'border-raidus': 'border-radius',
            r'text-allign': 'text-align'
        }
        
        for typo, correct in common_typos.items():
            if re.search(typo, css, re.IGNORECASE):
                result.warnings.append(f"Possible typo: {typo} (should be {correct})")


class JavaScriptValidator:
    """Validates generated JavaScript code"""
    
    def __init__(self):
        self.max_file_size = 1000000  # 1MB
    
    def validate(self, js: str) -> ValidationResult:
        """Validate JavaScript code"""
        result = ValidationResult(is_valid=True, code_type="javascript")
        
        try:
            # Check size
            if len(js) > self.max_file_size:
                result.errors.append(
                    f"JS size {len(js)} bytes exceeds limit {self.max_file_size}"
                )
            
            # Check if empty
            if not js.strip():
                result.warnings.append("Empty JavaScript")
                return result
            
            # Check for balanced braces/parentheses
            self._check_balance(js, result)
            
            # Check for security issues
            self._check_security_issues(js, result)
            
            # Check for common issues
            self._check_deprecated_features(js, result)
            self._check_syntax_issues(js, result)
            
            # Count functions and statements
            functions = len(re.findall(r'\bfunction\b|\b(const|let|var)\s+\w+\s*=\s*(?:async\s*)?\(', js))
            result.metrics['function_count'] = functions
            
            # Calculate score
            result.score = max(0, 1.0 - len(result.errors) * 0.3 - len(result.warnings) * 0.05)
            result.is_valid = len(result.errors) == 0
            
        except Exception as e:
            logger.error(f"JavaScript validation error: {e}")
            result.errors.append(str(e))
            result.is_valid = False
        
        return result
    
    def _check_balance(self, js: str, result: ValidationResult):
        """Check for balanced braces and parentheses"""
        braces = js.count('{') - js.count('}')
        parens = js.count('(') - js.count(')')
        brackets = js.count('[') - js.count(']')
        
        if braces != 0:
            result.errors.append(f"Unbalanced braces (balance: {braces})")
        if parens != 0:
            result.errors.append(f"Unbalanced parentheses (balance: {parens})")
        if brackets != 0:
            result.warnings.append(f"Unbalanced brackets (balance: {brackets})")
    
    def _check_security_issues(self, js: str, result: ValidationResult):
        """Check for common security issues"""
        security_risks = {
            r'eval\s*\(': 'eval() usage - security risk',
            r'innerHTML\s*=': 'innerHTML usage - XSS risk',
            r'document\.write': 'document.write() - should use DOM methods',
            r'with\s*\(': 'with statement - deprecated',
        }
        
        for pattern, risk in security_risks.items():
            if re.search(pattern, js):
                result.warnings.append(f"Security concern: {risk}")
    
    def _check_deprecated_features(self, js: str, result: ValidationResult):
        """Check for deprecated features"""
        if 'var ' in js:
            result.warnings.append("Using 'var' - prefer 'const' or 'let'")
        
        if re.search(r'function\s+\w+\s*\(.*\)\s*\{', js):
            result.metrics['function_declarations'] = len(
                re.findall(r'function\s+\w+\s*\(', js)
            )
    
    def _check_syntax_issues(self, js: str, result: ValidationResult):
        """Check for common syntax issues"""
        # Check for missing semicolons (basic check)
        lines = js.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if line and not line.endswith((';', '{', '}', ',', '://')):
                if not re.search(r'(?://|/\*)', line):  # Skip comments
                    if len(result.warnings) < 5:
                        result.warnings.append(f"Line {i+1}: possible missing semicolon")


class CodeConsistencyValidator:
    """Validates consistency between HTML, CSS, and JavaScript"""
    
    def validate(
        self,
        html: str,
        css: str,
        js: str
    ) -> ValidationResult:
        """Validate cross-file consistency"""
        result = ValidationResult(is_valid=True, code_type="combined")
        
        try:
            # Extract class names from HTML
            html_classes = set(re.findall(r'class=["\']([^"\']+)["\']', html))
            html_ids = set(re.findall(r'id=["\']([^"\']+)["\']', html))
            
            # Extract CSS selectors
            css_classes = set(re.findall(r'\.([a-zA-Z_][a-zA-Z0-9_-]*)', css))
            css_ids = set(re.findall(r'#([a-zA-Z_][a-zA-Z0-9_-]*)', css))
            
            # Extract JavaScript references
            js_classes = set(re.findall(r'["\']\.([a-zA-Z_][a-zA-Z0-9_-]*)["\']', js))
            js_ids = set(re.findall(r'["\']#([a-zA-Z_][a-zA-Z0-9_-]*)["\']', js))
            
            # Check for unused styles
            unused_classes = css_classes - html_classes - js_classes
            if unused_classes:
                unused_list = ', '.join(sorted(list(unused_classes))[:5])
                result.warnings.append(f"Unused CSS classes: {unused_list}")
            
            # Check for undefined styles
            undefined_classes = html_classes - css_classes
            if undefined_classes and len(undefined_classes) < 20:  # Limit check
                undefined_list = ', '.join(sorted(list(undefined_classes))[:3])
                result.warnings.append(f"No CSS defined for classes: {undefined_list}")
            
            # Check for mismatched JavaScript references
            missing_refs = (js_classes | js_ids) - (html_classes | html_ids)
            if missing_refs:
                missing_list = ', '.join(sorted(list(missing_refs))[:3])
                result.warnings.append(f"JS references undefined elements: {missing_list}")
            
            result.score = max(0, 1.0 - len(result.warnings) * 0.1)
            result.is_valid = True  # Consistency issues are warnings, not errors
            
        except Exception as e:
            logger.error(f"Consistency validation error: {e}")
            result.errors.append(str(e))
            result.is_valid = False
        
        return result


class CodeGeneratorValidator:
    """Main validator combining all checks"""
    
    def __init__(self):
        self.html_validator = HTMLValidator()
        self.css_validator = CSSValidator()
        self.js_validator = JavaScriptValidator()
        self.consistency_validator = CodeConsistencyValidator()
    
    def validate_all(
        self,
        html: str,
        css: str,
        js: str
    ) -> Dict[str, Any]:
        """Validate all code and return comprehensive report"""
        
        results = {
            'timestamp': str(__import__('datetime').datetime.utcnow().isoformat()),
            'html': self.html_validator.validate(html).to_dict(),
            'css': self.css_validator.validate(css).to_dict(),
            'js': self.js_validator.validate(js).to_dict(),
            'consistency': self.consistency_validator.validate(html, css, js).to_dict(),
        }
        
        # Calculate overall score
        scores = [
            results['html']['score'],
            results['css']['score'],
            results['js']['score'],
            results['consistency']['score']
        ]
        
        results['overall_score'] = sum(scores) / len(scores)
        results['is_valid'] = all(r['is_valid'] for r in [
            results['html'],
            results['css'],
            results['js']
        ])
        results['should_use'] = results['overall_score'] >= 0.7
        
        return results
    
    def validate_html(self, html: str) -> Dict[str, Any]:
        """Validate only HTML"""
        return self.html_validator.validate(html).to_dict()
    
    def validate_css(self, css: str) -> Dict[str, Any]:
        """Validate only CSS"""
        return self.css_validator.validate(css).to_dict()
    
    def validate_js(self, js: str) -> Dict[str, Any]:
        """Validate only JavaScript"""
        return self.js_validator.validate(js).to_dict()


# Example usage
if __name__ == "__main__":
    validator = CodeGeneratorValidator()
    
    # Example HTML
    html = """
    <html>
    <head><title>Test</title></head>
    <body>
        <div class="container">
            <h1>Hello</h1>
            <p>World</p>
        </div>
    </body>
    </html>
    """
    
    # Example CSS
    css = """
    .container {
        width: 1200px;
        margin: 0 auto;
    }
    h1 {
        color: #333;
        font-size: 32px;
    }
    """
    
    # Example JS
    js = """
    document.addEventListener('DOMContentLoaded', function() {
        const container = document.querySelector('.container');
        console.log('Page loaded');
    });
    """
    
    # Validate
    report = validator.validate_all(html, css, js)
    print(json.dumps(report, indent=2))
