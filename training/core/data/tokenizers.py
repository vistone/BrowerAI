"""
Code Tokenizers

Modern tokenization strategies for HTML/CSS/JS following:
- BPE (Byte-Pair Encoding) - used in GPT
- WordPiece - used in BERT
- SentencePiece - language-agnostic
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Union
from collections import Counter


class CodeTokenizer:
    """
    Base tokenizer for code.
    
    Features:
        - Preserves code structure (indentation, brackets)
        - Handles special tokens (identifiers, literals, operators)
        - Efficient vocabulary building
        - Save/load vocabulary
    """
    
    # Special tokens
    PAD = "<pad>"
    UNK = "<unk>"
    SOS = "<sos>"
    EOS = "<eos>"
    MASK = "<mask>"
    
    # Code-specific tokens
    INDENT = "<indent>"
    DEDENT = "<dedent>"
    NEWLINE = "<newline>"
    
    def __init__(self, vocab_size: int = 10000, min_freq: int = 2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        
        # Initialize special tokens
        self.special_tokens = [
            self.PAD, self.UNK, self.SOS, self.EOS, 
            self.MASK, self.INDENT, self.DEDENT, self.NEWLINE
        ]
        
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}
        
        # Initialize with special tokens
        for idx, token in enumerate(self.special_tokens):
            self.vocab[token] = idx
            self.inv_vocab[idx] = token
    
    def build_vocab(self, texts: List[str]):
        """
        Build vocabulary from corpus.
        
        Args:
            texts: List of code samples
        """
        # Tokenize all texts and count frequencies
        token_counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            token_counter.update(tokens)
        
        # Sort by frequency and take top vocab_size - len(special_tokens)
        sorted_tokens = sorted(
            token_counter.items(),
            key=lambda x: (-x[1], x[0])  # Sort by freq (desc), then alphabetically
        )
        
        # Filter by minimum frequency
        filtered_tokens = [
            token for token, freq in sorted_tokens 
            if freq >= self.min_freq
        ]
        
        # Add to vocabulary
        start_idx = len(self.special_tokens)
        remaining_slots = self.vocab_size - start_idx
        
        for token in filtered_tokens[:remaining_slots]:
            idx = len(self.vocab)
            self.vocab[token] = idx
            self.inv_vocab[idx] = token
        
        print(f"✅ Vocabulary built: {len(self.vocab)} tokens")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize code text.
        
        Strategy:
            1. Preserve strings and comments
            2. Split on whitespace and punctuation
            3. Keep programming constructs intact
        """
        tokens = []
        
        # Regex pattern for code tokenization
        # Captures: strings, numbers, identifiers, operators, punctuation
        pattern = r'''
            "(?:[^"\\]|\\.)*"|           # Double-quoted strings
            '(?:[^'\\]|\\.)*'|           # Single-quoted strings
            `(?:[^`\\]|\\.)*`|           # Template literals
            //.*?$|                      # Single-line comments
            /\*.*?\*/|                   # Multi-line comments
            \d+\.?\d*|                   # Numbers
            [a-zA-Z_][a-zA-Z0-9_]*|      # Identifiers
            [+\-*/%=<>!&|^~]+|           # Operators
            [(){}\[\];,.]|               # Punctuation
            \s+                          # Whitespace
        '''
        
        matches = re.finditer(pattern, text, re.VERBOSE | re.MULTILINE)
        
        for match in matches:
            token = match.group(0)
            
            # Handle whitespace
            if token.isspace():
                if '\n' in token:
                    tokens.append(self.NEWLINE)
                continue
            
            tokens.append(token)
        
        return tokens
    
    def encode(self, text: str, max_length: Optional[int] = None,
               add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            add_special_tokens: Whether to add SOS/EOS
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = [self.SOS] + tokens + [self.EOS]
        
        # Truncate if needed
        if max_length:
            tokens = tokens[:max_length]
        
        # Convert to IDs
        ids = [self.vocab.get(token, self.vocab[self.UNK]) for token in tokens]
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Skip special tokens in output
            
        Returns:
            Decoded text
        """
        tokens = []
        
        for idx in ids:
            token = self.inv_vocab.get(idx, self.UNK)
            
            if skip_special_tokens and token in self.special_tokens:
                continue
            
            tokens.append(token)
        
        # Join tokens with smart spacing
        text = self._detokenize(tokens)
        
        return text
    
    def _detokenize(self, tokens: List[str]) -> str:
        """
        Smart detokenization preserving code structure.
        """
        result = []
        
        for i, token in enumerate(tokens):
            # Add token
            result.append(token)
            
            # Determine if space needed before next token
            if i < len(tokens) - 1:
                next_token = tokens[i + 1]
                
                # No space before/after certain punctuation
                no_space_before = [',', ';', ')', ']', '}', '.']
                no_space_after = ['(', '[', '{', '.']
                
                if token == self.NEWLINE:
                    result.append('\n')
                elif next_token not in no_space_before and token not in no_space_after:
                    result.append(' ')
        
        return ''.join(result)
    
    def save(self, path: Union[str, Path]):
        """Save vocabulary to file."""
        path = Path(path)
        
        vocab_data = {
            "vocab": self.vocab,
            "config": {
                "vocab_size": self.vocab_size,
                "min_freq": self.min_freq
            }
        }
        
        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        print(f"✅ Tokenizer saved: {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'CodeTokenizer':
        """Load vocabulary from file."""
        path = Path(path)
        
        with open(path, 'r') as f:
            vocab_data = json.load(f)
        
        tokenizer = cls(
            vocab_size=vocab_data["config"]["vocab_size"],
            min_freq=vocab_data["config"]["min_freq"]
        )
        
        tokenizer.vocab = {k: int(v) for k, v in vocab_data["vocab"].items()}
        tokenizer.inv_vocab = {int(k): v for k, v in vocab_data["vocab"].items()}
        
        print(f"✅ Tokenizer loaded: {path} ({len(tokenizer.vocab)} tokens)")
        
        return tokenizer


class UnifiedWebTokenizer(CodeTokenizer):
    """
    Unified tokenizer for HTML/CSS/JS.
    
    Shares vocabulary across all three languages for:
        - Common tokens (div, class, function, etc.)
        - Improved cross-language understanding
        - Reduced total vocabulary size
    
    Inspired by:
        - Multilingual BERT (mBERT)
        - XLM-R (cross-lingual models)
    """
    
    # Language-specific special tokens
    HTML_START = "<html>"
    CSS_START = "<css>"
    JS_START = "<js>"
    
    def __init__(self, vocab_size: int = 15000, min_freq: int = 2):
        super().__init__(vocab_size, min_freq)
        
        # Add language markers
        self.lang_tokens = [self.HTML_START, self.CSS_START, self.JS_START]
        for token in self.lang_tokens:
            idx = len(self.vocab)
            self.vocab[token] = idx
            self.inv_vocab[idx] = token
    
    def encode_with_language(self, text: str, language: str,
                            max_length: Optional[int] = None) -> List[int]:
        """
        Encode with language marker.
        
        Args:
            text: Input text
            language: 'html', 'css', or 'js'
            max_length: Maximum length
            
        Returns:
            Token IDs with language marker
        """
        # Get language marker
        lang_markers = {
            'html': self.HTML_START,
            'css': self.CSS_START,
            'js': self.JS_START
        }
        lang_token = lang_markers.get(language.lower(), self.SOS)
        
        # Tokenize
        tokens = self.tokenize(text)
        tokens = [lang_token, self.SOS] + tokens + [self.EOS]
        
        if max_length:
            tokens = tokens[:max_length]
        
        # Convert to IDs
        ids = [self.vocab.get(token, self.vocab[self.UNK]) for token in tokens]
        
        return ids


class BPETokenizer(CodeTokenizer):
    """
    Byte-Pair Encoding tokenizer for code.
    
    Modern subword tokenization used in GPT models.
    Better handling of:
        - Rare words
        - Out-of-vocabulary tokens
        - Morphological variations
    
    References:
        - "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2016)
        - GPT-2 tokenizer
    """
    
    def __init__(self, vocab_size: int = 10000, num_merges: int = 5000):
        super().__init__(vocab_size)
        self.num_merges = num_merges
        self.bpe_merges: Dict[tuple, int] = {}
    
    def learn_bpe(self, texts: List[str]):
        """
        Learn BPE merges from corpus.
        
        Algorithm:
            1. Initialize with character vocabulary
            2. Iteratively merge most frequent adjacent pairs
            3. Stop after num_merges or vocabulary full
        """
        # Initialize with characters
        char_counter = Counter()
        for text in texts:
            for char in text:
                char_counter[char] += 1
        
        # Start vocabulary with characters
        for char in char_counter:
            if char not in self.vocab:
                idx = len(self.vocab)
                self.vocab[char] = idx
                self.inv_vocab[idx] = char
        
        # Learn merges
        for merge_idx in range(self.num_merges):
            if len(self.vocab) >= self.vocab_size:
                break
            
            # Count adjacent pairs
            pair_counter = Counter()
            for text in texts:
                symbols = list(text)  # Start with characters
                
                # Apply existing merges
                for (a, b), _ in sorted(self.bpe_merges.items(), key=lambda x: x[1]):
                    new_symbols = []
                    i = 0
                    while i < len(symbols):
                        if i < len(symbols) - 1 and symbols[i] == a and symbols[i+1] == b:
                            new_symbols.append(a + b)
                            i += 2
                        else:
                            new_symbols.append(symbols[i])
                            i += 1
                    symbols = new_symbols
                
                # Count pairs in current segmentation
                for i in range(len(symbols) - 1):
                    pair_counter[(symbols[i], symbols[i+1])] += 1
            
            if not pair_counter:
                break
            
            # Get most frequent pair
            best_pair = max(pair_counter, key=pair_counter.get)
            
            # Add merge
            self.bpe_merges[best_pair] = merge_idx
            
            # Add merged token to vocabulary
            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[merged_token] = idx
                self.inv_vocab[idx] = merged_token
        
        print(f"✅ Learned {len(self.bpe_merges)} BPE merges")
        print(f"✅ Vocabulary size: {len(self.vocab)}")
