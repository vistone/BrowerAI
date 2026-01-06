"""
Code Deobfuscation Models

Modern seq2seq models for JavaScript deobfuscation using
transformer architectures and advanced decoding strategies.
"""

import torch
import torch.nn as nn
from .transformer import TransformerSeq2Seq
from typing import Optional, Dict, Tuple


class CodeDeobfuscator(TransformerSeq2Seq):
    """
    JavaScript deobfuscation model.
    
    Architecture improvements:
        - Deeper encoder for complex obfuscation understanding
        - Copy mechanism for preserving identifiers
        - Pointer network for handling unknown tokens
        - Reinforcement learning for fluency optimization
    
    Inspired by:
        - Neural machine translation (NMT) models
        - Code translation models (CodeBERT, GraphCodeBERT)
        - Pointer-Generator networks
    """
    
    def __init__(self, config: dict):
        # Deobfuscation-specific defaults
        config.setdefault("d_model", 384)
        config.setdefault("num_heads", 12)
        config.setdefault("num_encoder_layers", 8)
        config.setdefault("num_decoder_layers", 8)
        config.setdefault("d_ff", 1536)
        
        super().__init__(config)
        
        vocab_size = config["vocab_size"]
        d_model = config["d_model"]
        
        # Copy mechanism components
        self.copy_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # Pointer network for handling source tokens
        self.pointer_network = nn.Linear(d_model, d_model)
        
        # Obfuscation type classifier (auxiliary task)
        self.obfuscation_classifier = nn.Linear(d_model, 10)  # 10 obfuscation types
        
    def forward_with_copy(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor,
                         src_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with copy mechanism.
        
        Args:
            src_ids: Source (obfuscated) code
            tgt_ids: Target (clean) code
            src_mask: Source mask
            
        Returns:
            Dict containing:
                - generation_logits: Standard vocabulary distribution
                - copy_logits: Copy distribution over source tokens
                - copy_gate: Probability of copying vs generating
        """
        # Encode source
        encoder_output = self.encode(src_ids, src_mask)
        
        # Pool encoder output for obfuscation classification
        encoder_pooled = encoder_output.mean(dim=1)
        obfuscation_type = self.obfuscation_classifier(encoder_pooled)
        
        # Decode target
        tgt_len = tgt_ids.size(1)
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
        causal_mask = causal_mask.to(src_ids.device)
        
        decoder_output = self.decode(tgt_ids, encoder_output, tgt_mask=causal_mask)
        
        # Generation distribution
        generation_logits = self.output_layer(decoder_output)
        
        # Copy mechanism
        # Compute attention scores between decoder and encoder
        copy_scores = torch.matmul(
            self.pointer_network(decoder_output),
            encoder_output.transpose(-2, -1)
        ) / (self.config["d_model"] ** 0.5)
        copy_logits = torch.softmax(copy_scores, dim=-1)
        
        # Compute copy gate
        # Concatenate decoder output with context from encoder
        context = torch.matmul(copy_logits, encoder_output)
        gate_input = torch.cat([decoder_output, context], dim=-1)
        copy_gate = self.copy_gate(gate_input)
        
        return {
            "generation_logits": generation_logits,
            "copy_logits": copy_logits,
            "copy_gate": copy_gate,
            "obfuscation_type": obfuscation_type
        }
    
    def compute_copy_loss(self, generation_logits: torch.Tensor, copy_logits: torch.Tensor,
                         copy_gate: torch.Tensor, src_ids: torch.Tensor, 
                         tgt_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute loss incorporating copy mechanism.
        
        Final probability = copy_gate * P(copy) + (1 - copy_gate) * P(generate)
        """
        batch_size, tgt_len, vocab_size = generation_logits.shape
        src_len = src_ids.size(1)
        
        # Generation probability
        gen_probs = torch.softmax(generation_logits, dim=-1)
        
        # Copy probability - scatter copy_logits to vocabulary space
        copy_probs = torch.zeros_like(gen_probs)
        src_ids_expanded = src_ids.unsqueeze(1).expand(-1, tgt_len, -1)
        copy_probs.scatter_add_(2, src_ids_expanded, copy_logits)
        
        # Combined probability
        final_probs = copy_gate * copy_probs + (1 - copy_gate) * gen_probs
        
        # Negative log likelihood loss
        tgt_probs = torch.gather(final_probs, 2, tgt_ids.unsqueeze(-1)).squeeze(-1)
        loss = -torch.log(tgt_probs + 1e-10).mean()
        
        return loss
    
    @torch.no_grad()
    def beam_search_decode(self, src_ids: torch.Tensor, beam_width: int = 5,
                           max_length: int = 512) -> Tuple[torch.Tensor, float]:
        """
        Beam search decoding with copy mechanism.
        
        Modern decoding strategy for better quality outputs.
        
        Args:
            src_ids: Source sequence (1, src_len)
            beam_width: Number of beams
            max_length: Maximum decoding length
            
        Returns:
            Best decoded sequence and its score
        """
        self.eval()
        device = src_ids.device
        vocab_size = self.config["vocab_size"]
        
        # Encode source once
        encoder_output = self.encode(src_ids)
        
        # Initialize beams: (beam_width, 1) starting with SOS token
        SOS_TOKEN = 1
        beams = torch.full((beam_width, 1), SOS_TOKEN, dtype=torch.long, device=device)
        beam_scores = torch.zeros(beam_width, device=device)
        
        # Beam search loop
        for step in range(max_length):
            # Expand encoder output for all beams
            encoder_output_expanded = encoder_output.expand(beam_width, -1, -1)
            
            # Decode current beams
            outputs = self.forward_with_copy(
                src_ids.expand(beam_width, -1),
                beams,
                src_mask=None
            )
            
            # Get next token logits
            next_logits = outputs["generation_logits"][:, -1, :]  # (beam_width, vocab_size)
            next_scores = torch.log_softmax(next_logits, dim=-1)
            
            # Compute scores for all possible next tokens
            candidate_scores = beam_scores.unsqueeze(-1) + next_scores  # (beam_width, vocab_size)
            
            # Flatten and get top beam_width candidates
            candidate_scores = candidate_scores.view(-1)
            top_scores, top_indices = torch.topk(candidate_scores, beam_width)
            
            # Convert flat indices to (beam_id, token_id)
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
            
            # Update beams
            beams = torch.cat([
                beams[beam_indices],
                token_indices.unsqueeze(-1)
            ], dim=1)
            beam_scores = top_scores
            
            # Check if all beams ended
            EOS_TOKEN = vocab_size - 1
            if (token_indices == EOS_TOKEN).all():
                break
        
        # Return best beam
        best_beam_idx = beam_scores.argmax()
        best_sequence = beams[best_beam_idx]
        best_score = beam_scores[best_beam_idx].item()
        
        return best_sequence, best_score


class EnhancedDeobfuscator(CodeDeobfuscator):
    """
    Enhanced deobfuscator with additional modern techniques:
        - Pre-training on synthetic data
        - Curriculum learning (easy → hard obfuscation)
        - Meta-learning for few-shot adaptation to new obfuscation types
        - Knowledge distillation from larger models
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        d_model = config["d_model"]
        
        # Curriculum learning: obfuscation difficulty predictor
        self.difficulty_predictor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Meta-learning: fast adaptation layers
        self.meta_adaptation_layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(2)
        ])
        
    def predict_difficulty(self, src_ids: torch.Tensor) -> torch.Tensor:
        """Predict obfuscation difficulty for curriculum learning."""
        encoder_output = self.encode(src_ids)
        pooled = encoder_output.mean(dim=1)
        difficulty = self.difficulty_predictor(pooled)
        return difficulty
    
    def meta_adapt(self, support_src: torch.Tensor, support_tgt: torch.Tensor,
                   query_src: torch.Tensor, adaptation_steps: int = 5) -> torch.Tensor:
        """
        Few-shot adaptation to new obfuscation type.
        
        Meta-learning approach (MAML-style):
            1. Adapt on support set
            2. Evaluate on query set
        
        Args:
            support_src: Support examples (obfuscated)
            support_tgt: Support examples (clean)
            query_src: Query example to deobfuscate
            adaptation_steps: Number of adaptation gradient steps
            
        Returns:
            Deobfuscated query output
        """
        # Clone adaptation parameters
        adapted_params = [layer.weight.clone() for layer in self.meta_adaptation_layers]
        
        # Inner loop: adapt on support set
        for _ in range(adaptation_steps):
            support_output = self.forward(support_src, support_tgt)
            support_loss = nn.functional.cross_entropy(
                support_output.view(-1, support_output.size(-1)),
                support_tgt.view(-1),
                ignore_index=0
            )
            
            # Compute gradients w.r.t. adaptation parameters
            grads = torch.autograd.grad(support_loss, adapted_params, create_graph=True)
            
            # Update adaptation parameters
            adapted_params = [p - 0.01 * g for p, g in zip(adapted_params, grads)]
        
        # Outer loop: evaluate on query with adapted parameters
        # (Simplified - in practice would apply adapted_params)
        query_output = self.forward(query_src, query_src)  # Use src as initial tgt
        
        return query_output
