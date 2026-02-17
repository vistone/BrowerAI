"""
Code Generator Module - Generates HTML/CSS/JS from latent vectors
Converts 256-dimensional latent representations into web code
"""

import numpy as np
from typing import Dict, Any, List
import logging
import random

logger = logging.getLogger(__name__)


class CodeGenerator:
    """
    Generates HTML/CSS/JavaScript from latent vectors
    Uses templates and latent-guided code synthesis
    """
    
    def __init__(self, latent_dim: int = 256):
        """
        Initialize code generator
        
        Args:
            latent_dim: Latent vector dimension (256)
        """
        self.latent_dim = latent_dim
        self.generation_count = 0
        
        # Code templates database
        self.html_templates = self._init_html_templates()
        self.css_templates = self._init_css_templates()
        self.js_templates = self._init_js_templates()
        
        logger.info(f"CodeGenerator initialized with {latent_dim}-dim latent space")
    
    def generate(
        self,
        latent_vector: np.ndarray,
        session_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Generate HTML/CSS/JS from latent vector
        
        Args:
            latent_vector: 256-dimensional latent representation
            session_id: Session identifier for tracking
        
        Returns:
            Dictionary with generated code and metadata
        """
        try:
            if len(latent_vector) != self.latent_dim:
                raise ValueError(
                    f"Expected {self.latent_dim}-dim latent vector, "
                    f"got {len(latent_vector)}"
                )
            
            # Derive generation parameters from latent vector
            params = self._latent_to_params(latent_vector)
            
            # Generate HTML
            html = self._generate_html(params)
            
            # Generate CSS
            css = self._generate_css(params)
            
            # Generate JavaScript
            javascript = self._generate_javascript(params)
            
            # Calculate confidence based on latent vector properties
            confidence = self._calculate_confidence(latent_vector)
            
            # Compute loss (for training feedback)
            loss = 1.0 - confidence  # Simple loss: opposite of confidence
            
            self.generation_count += 1
            
            logger.info(
                f"Generated code (session={session_id}): "
                f"confidence={confidence:.3f}, loss={loss:.3f}"
            )
            
            return {
                "html": html,
                "css": css,
                "javascript": javascript,
                "confidence": float(confidence),
                "loss": float(loss),
                "epoch": self.generation_count,
                "session_id": session_id,
            }
        
        except Exception as e:
            logger.error(f"Code generation error: {e}")
            raise
    
    def _latent_to_params(self, latent_vector: np.ndarray) -> Dict[str, Any]:
        """Convert latent vector to generation parameters"""
        
        # Normalize latent vector to [0, 1]
        latent_norm = (latent_vector - latent_vector.min()) / (
            latent_vector.max() - latent_vector.min() + 1e-6
        )
        
        # Extract parameters
        return {
            "layout_type": self._select_layout(latent_norm[:10]),
            "color_scheme": self._select_colors(latent_norm[10:20]),
            "typography": self._select_typography(latent_norm[20:40]),
            "spacing": self._select_spacing(latent_norm[40:100]),
            "animations": self._select_animations(latent_norm[100:150]),
            "complexity": self._select_complexity(latent_norm[150:]),
            "latent_mean": float(latent_norm.mean()),
            "latent_std": float(latent_norm.std()),
        }
    
    def _select_layout(self, values: np.ndarray) -> str:
        """Select layout type from latent values"""
        layouts = [
            "single-column",
            "two-column",
            "grid",
            "masonry",
            "flexbox",
            "sidebar-main"
        ]
        idx = int(values[0] * len(layouts)) % len(layouts)
        return layouts[idx]
    
    def _select_colors(self, values: np.ndarray) -> Dict[str, str]:
        """Select color scheme from latent values"""
        color_palettes = [
            {"primary": "#3498db", "secondary": "#2ecc71", "accent": "#e74c3c"},
            {"primary": "#9b59b6", "secondary": "#f39c12", "accent": "#1abc9c"},
            {"primary": "#34495e", "secondary": "#95a5a6", "accent": "#c0392b"},
            {"primary": "#16a085", "secondary": "#27ae60", "accent": "#f1c40f"},
            {"primary": "#2980b9", "secondary": "#e67e22", "accent": "#ecf0f1"},
        ]
        idx = int(values[0] * len(color_palettes)) % len(color_palettes)
        return color_palettes[idx]
    
    def _select_typography(self, values: np.ndarray) -> Dict[str, Any]:
        """Select typography from latent values"""
        fonts = [
            {"body": "Segoe UI, sans-serif", "heading": "Arial, sans-serif"},
            {"body": "Helvetica, sans-serif", "heading": "Georgia, serif"},
            {"body": "Verdana, sans-serif", "heading": "Times, serif"},
            {"body": "Courier, monospace", "heading": "Comic Sans, cursive"},
            {"body": "Trebuchet MS, sans-serif", "heading": "Impact, sans-serif"},
        ]
        idx = int(values[0] * len(fonts)) % len(fonts)
        font_size = 12 + int(values[1] * 8)
        
        return {
            **fonts[idx],
            "base_size": f"{font_size}px",
            "line_height": 1.5 + values[2] * 0.5,
        }
    
    def _select_spacing(self, values: np.ndarray) -> Dict[str, str]:
        """Select spacing/padding from latent values"""
        spacing_scales = [
            {"small": "4px", "medium": "8px", "large": "16px"},
            {"small": "8px", "medium": "16px", "large": "32px"},
            {"small": "12px", "medium": "24px", "large": "48px"},
        ]
        idx = int(values[0] * len(spacing_scales)) % len(spacing_scales)
        return spacing_scales[idx]
    
    def _select_animations(self, values: np.ndarray) -> List[str]:
        """Select animations from latent values"""
        all_animations = [
            "fade", "slide", "bounce", "scale", "rotate",
            "pulse", "glow", "hover-lift", "smooth-scroll"
        ]
        
        count = int(values[0] * 5) + 1
        selected = random.sample(all_animations, min(count, len(all_animations)))
        
        return selected
    
    def _select_complexity(self, values: np.ndarray) -> str:
        """Select complexity level from latent values"""
        complexity_levels = ["simple", "moderate", "complex"]
        idx = int(values[0] * len(complexity_levels)) % len(complexity_levels)
        return complexity_levels[idx]
    
    def _generate_html(self, params: Dict[str, Any]) -> str:
        """Generate HTML structure"""
        layout = params["layout_type"]
        complexity = params["complexity"]
        
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generated Website</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <header class="header">
        <nav class="navbar">
            <div class="logo">BrowserAI</div>
            <ul class="nav-links">
                <li><a href="#home">Home</a></li>
                <li><a href="#about">About</a></li>
                <li><a href="#services">Services</a></li>
                <li><a href="#contact">Contact</a></li>
            </ul>
        </nav>
    </header>
    
    <main class="container" data-layout="{layout}">"""
        
        # Add content sections based on complexity
        if complexity == "simple":
            html += """
        <section class="hero">
            <h1>Welcome</h1>
            <p>Generated by BrowserAI Learning System</p>
        </section>
        <section class="content">
            <h2>Featured Content</h2>
            <p>Automatically generated based on website analysis</p>
        </section>"""
        
        elif complexity == "moderate":
            html += """
        <section class="hero">
            <h1>Welcome to Our Site</h1>
            <p>Explore our AI-generated content</p>
        </section>
        <section class="features">
            <article><h3>Feature 1</h3><p>Description</p></article>
            <article><h3>Feature 2</h3><p>Description</p></article>
            <article><h3>Feature 3</h3><p>Description</p></article>
        </section>
        <section class="content">
            <h2>Main Content</h2>
            <p>Rich content generated from features</p>
            <ul>
                <li>Point 1</li>
                <li>Point 2</li>
                <li>Point 3</li>
            </ul>
        </section>"""
        
        else:  # complex
            html += """
        <section class="hero">
            <h1>Welcome</h1>
            <p>Comprehensive AI-Generated Website</p>
        </section>
        <section class="features" data-columns="3">
            <article><h3>Advanced Feature 1</h3><p>Detailed description</p></article>
            <article><h3>Advanced Feature 2</h3><p>Detailed description</p></article>
            <article><h3>Advanced Feature 3</h3><p>Detailed description</p></article>
            <article><h3>Advanced Feature 4</h3><p>Detailed description</p></article>
            <article><h3>Advanced Feature 5</h3><p>Detailed description</p></article>
            <article><h3>Advanced Feature 6</h3><p>Detailed description</p></article>
        </section>
        <section class="testimonials">
            <h2>User Testimonials</h2>
            <div class="testimonial">
                <p>"Excellent experience"</p>
                <footer>- User 1</footer>
            </div>
            <div class="testimonial">
                <p>"Highly recommended"</p>
                <footer>- User 2</footer>
            </div>
        </section>
        <section class="content">
            <h2>Detailed Information</h2>
            <p>Comprehensive content generated from analysis</p>
        </section>"""
        
        html += """
    </main>
    
    <footer class="footer">
        <p>&copy; 2024 Generated by BrowserAI. All rights reserved.</p>
    </footer>
    
    <script src="script.js"></script>
</body>
</html>"""
        
        return html
    
    def _generate_css(self, params: Dict[str, Any]) -> str:
        """Generate CSS styling"""
        colors = params["color_scheme"]
        typography = params["typography"]
        spacing = params["spacing"]
        
        css = f"""/* Generated CSS */

* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

body {{
    font-family: {typography['body']};
    font-size: {typography['base_size']};
    line-height: {typography['line_height']};
    color: #333;
    background-color: #f9f9f9;
}}

header {{
    background-color: {colors['primary']};
    color: white;
    padding: {spacing['medium']};
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}

.navbar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    max-width: 1200px;
    margin: 0 auto;
}}

.logo {{
    font-size: 24px;
    font-weight: bold;
}}

.nav-links {{
    display: flex;
    list-style: none;
    gap: {spacing['large']};
}}

.nav-links a {{
    color: white;
    text-decoration: none;
    transition: opacity 0.3s ease;
}}

.nav-links a:hover {{
    opacity: 0.8;
}}

.container {{
    max-width: 1200px;
    margin: 0 auto;
    padding: {spacing['large']};
}}

section {{
    margin-bottom: {spacing['large']};
    padding: {spacing['medium']};
    background-color: white;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}}

h1, h2, h3 {{
    font-family: {typography['heading']};
    color: {colors['primary']};
    margin-bottom: {spacing['medium']};
}}

h1 {{
    font-size: 32px;
}}

h2 {{
    font-size: 24px;
}}

h3 {{
    font-size: 18px;
}}

article {{
    padding: {spacing['medium']};
    border-left: 4px solid {colors['secondary']};
    margin-bottom: {spacing['medium']};
}}

.features {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: {spacing['large']};
}}

footer {{
    background-color: {colors['primary']};
    color: white;
    text-align: center;
    padding: {spacing['medium']};
    margin-top: {spacing['large']};
}}

a {{
    color: {colors['secondary']};
    text-decoration: none;
}}

a:hover {{
    color: {colors['accent']};
}}

/* Animations */
@keyframes fadeIn {{
    from {{ opacity: 0; }}
    to {{ opacity: 1; }}
}}

section {{
    animation: fadeIn 0.5s ease-in;
}}
"""
        
        return css
    
    def _generate_javascript(self, params: Dict[str, Any]) -> str:
        """Generate JavaScript code"""
        animations = params["animations"]
        
        js = """// Generated JavaScript

document.addEventListener('DOMContentLoaded', function() {
    console.log('BrowserAI: Page loaded');
    initializeNavigation();
    initializeAnimations();
});

function initializeNavigation() {
    const navLinks = document.querySelectorAll('.nav-links a');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            if (targetId.startsWith('#')) {
                const target = document.querySelector(targetId);
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth' });
                }
            }
        });
    });
}

function initializeAnimations() {
    // Observe elements for animations
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animated');
            }
        });
    });
    
    document.querySelectorAll('section').forEach(section => {
        observer.observe(section);
    });
}

// Utility functions
function log(message) {
    console.log('[BrowserAI] ' + message);
}

function ready(fn) {
    if (document.readyState !== 'loading') {
        fn();
    } else {
        document.addEventListener('DOMContentLoaded', fn);
    }
}

// Export functions for use
window.BrowserAI = {
    initializeNavigation: initializeNavigation,
    initializeAnimations: initializeAnimations,
    log: log
};

console.log('BrowserAI: JavaScript initialized');
"""
        
        return js
    
    def _calculate_confidence(self, latent_vector: np.ndarray) -> float:
        """Calculate confidence score based on latent vector properties"""
        
        # Confidence based on vector properties
        magnitude = np.linalg.norm(latent_vector)
        magnitude_confidence = min(1.0, magnitude / np.sqrt(self.latent_dim))
        
        # Confidence based on variance
        variance = np.var(latent_vector)
        variance_confidence = 1.0 - np.exp(-variance)
        
        # Confidence based on sparsity (prefer non-sparse vectors)
        sparsity = np.mean(np.abs(latent_vector) < 0.1)
        sparsity_confidence = 1.0 - sparsity
        
        # Combined confidence
        confidence = (
            magnitude_confidence * 0.3 +
            variance_confidence * 0.4 +
            sparsity_confidence * 0.3
        )
        
        return float(np.clip(confidence, 0.5, 0.99))
    
    def _init_html_templates(self) -> List[str]:
        """Initialize HTML templates"""
        return [
            "standard",
            "modern",
            "minimal",
            "grid",
            "list"
        ]
    
    def _init_css_templates(self) -> List[str]:
        """Initialize CSS templates"""
        return [
            "bootstrap",
            "tailwind",
            "bulma",
            "foundation",
            "custom"
        ]
    
    def _init_js_templates(self) -> List[str]:
        """Initialize JavaScript templates"""
        return [
            "vanilla",
            "jquery",
            "react-style",
            "interactive",
            "enhanced"
        ]
