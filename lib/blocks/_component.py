import torch
import torch.nn as nn
import torch.nn.functional as F
import math


####################
#### Layer Norm ####
####################


class LayerNorm(nn.Module):
    """RMSNorm without mean centering"""
    
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps    = eps
    
    def forward(self, x):
        variance    = x.pow(2).mean(-1, keepdim=True)
        x           = x * torch.rsqrt(variance + self.eps)

        return self.weight * x


###################
#### Attention ####
###################


class Attention(nn.Module):
    """ Class for Generalized Attention """
    
    def __init__(
        self,
        embed_dim           : int,
        num_heads           : int,
        dropout             : float = 0.0,
        is_decoder          : bool = False,
        is_cross_attention  : bool = False,
        has_relative_bias   : bool = True,
        use_alibi           : bool = False,
        use_flash           : bool = True
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert not (has_relative_bias and use_alibi), "Cannot use both relative bias and ALiBi"
        
        self.embed_dim          = embed_dim
        self.num_heads          = num_heads
        self.head_dim           = embed_dim // num_heads
        self.is_decoder         = is_decoder
        self.is_cross_attention = is_cross_attention
        self.has_relative_bias  = has_relative_bias
        self.use_alibi          = use_alibi
        self.use_flash          = use_flash
        
        # No Bias in Attention Projection
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Position bias
        if not is_cross_attention:
            if use_alibi:
                self.position_bias_module = ALiBi(num_heads=num_heads)
            elif has_relative_bias:
                self.position_bias_module = RelativePositionBias(
                    num_heads=num_heads,
                    bidirectional=not is_decoder
                )
            else:
                self.position_bias_module = None
        else:
            self.position_bias_module = None
    
    def forward(self, hidden_states, key_value_states=None, attention_mask=None, position_bias=None):
        """
        Args:
            hidden_states   : (B, L, D) query states
            key_value_states: (B, L_kv, D) for cross-attention, None for self-attention
            attention_mask  : (B, 1, L, L_kv) additive mask
            position_bias   : precomputed position bias for reuse
        
        Returns:
            output          : (B, L, D)
            position_bias   : for reuse in subsequent layers
        """
        
        B, L, D = hidden_states.shape
        
        # Query from hidden_states
        q = self.q(hidden_states)
        
        # k&v from cross-attn
        if self.is_cross_attention and key_value_states is not None:
            k       = self.k(key_value_states)
            v       = self.v(key_value_states)
            L_kv    = key_value_states.shape[1]
        # k&v from self-attn
        else:
            k       = self.k(hidden_states)
            v       = self.v(hidden_states)
            L_kv    = L
        
        # Reshape: (B, L, D) -> (B, num_heads, L, head_dim)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute position bias
        if self.position_bias_module is not None and position_bias is None:
            position_bias = self.position_bias_module(L, L_kv, hidden_states.device)
        
        # Use Flash Attention if available
        use_flash_attn = (
            self.use_flash 
            and hasattr(F, 'scaled_dot_product_attention')
            and position_bias is None 
            and attention_mask is None
        )
        
        if use_flash_attn:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=self.is_decoder and not self.is_cross_attention
            )
        else:
            # Compute Attention
            scores = torch.matmul(q, k.transpose(-2, -1))  # (B, num_heads, L, L_kv)
            
            # Apply relative position bias
            if position_bias is not None:
                scores = scores + position_bias
            
            # Apply attention mask
            if attention_mask is not None:
                scores = scores + attention_mask
            
            # Softmax and dropout
            attn_weights    = F.softmax(scores.float(), dim=-1).type_as(scores)
            attn_weights    = self.dropout(attn_weights)
            out             = torch.matmul(attn_weights, v)
        
        # Reshape back: (B, num_heads, L, head_dim) -> (B, L, D)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.o(out)
        
        return out, position_bias


##################################
#### BigBird Sparse Attention ####
##################################


class BigBirdSparseAttention(nn.Module):
    """
    BigBird sparse attention mechanism for encoder (bidirectional)
    
    Combines three attention patterns:
        1. Global tokens    : First/last blocks attend to all and are attended by all
        2. Window attention : Each token attends to local neighborhood
        3. Random attention : Each token attends to random set of blocks
    """
    
    def __init__(
        self,
        embed_dim           : int,
        num_heads           : int,
        block_size          : int = 64,
        num_rand_blocks     : int = 3,
        dropout             : float = 0.0,
        use_alibi           : bool = False,
        use_flash           : bool = False
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim          = embed_dim
        self.num_heads          = num_heads
        self.head_dim           = embed_dim // num_heads
        self.block_size         = block_size
        self.num_rand_blocks    = num_rand_blocks
        self.use_alibi          = use_alibi
        self.use_flash          = use_flash
        
        # Attention uses 3 blocks
        self.window_size = 3
        
        # Global blocks
        self.num_global_blocks = 2
        
        # Projections
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # ALiBi position bias
        if use_alibi:
            self.alibi = ALiBi(num_heads=num_heads)
        else:
            self.alibi = None
        
        # Initialize random seed
        self._seed = None
    
    def _get_random_block_indices(self, num_blocks, device):
        """ Generate random block indices for each non-global block """
        
        # Exclude global blocks
        num_non_global = num_blocks - self.num_global_blocks
        
        # Each non-global block sample random blocks
        rand_indices = []
        for i in range(num_non_global):
            # Can attend to any block except itself (block i+1 in full sequence)
            block_idx = i + 1  # +1 because first block is global
            
            # Available blocks to sample from (exclude self)
            available = list(range(num_blocks))
            available.remove(block_idx)
            
            # Randomly sample num_rand_blocks
            if self._seed is not None:
                torch.manual_seed(self._seed + i)
            
            sampled         = torch.randperm(len(available), device=device)[:self.num_rand_blocks]
            sampled_blocks  = torch.tensor([available[idx] for idx in sampled], device=device)
            rand_indices.append(sampled_blocks)
        
        return torch.stack(rand_indices)
    
    def _create_sparse_attention_mask(self, seq_len, device):
        """ Create BigBird sparse attention mask """
        
        num_blocks = seq_len // self.block_size
        
        # Initialize mask as all False (no attention)
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        
        # Global attention
        mask[:self.block_size, :]   = True
        mask[:, :self.block_size]   = True
        mask[-self.block_size:, :]  = True
        mask[:, -self.block_size:]  = True
        
        # Window attention
        for block_idx in range(1, num_blocks - 1):
            start_pos = block_idx * self.block_size
            end_pos = start_pos + self.block_size
            
            # Attend to previous block
            if block_idx > 0:
                prev_start = (block_idx - 1) * self.block_size
                prev_end = prev_start + self.block_size
                mask[start_pos:end_pos, prev_start:prev_end] = True
            
            # Attend to self
            mask[start_pos:end_pos, start_pos:end_pos] = True
            
            # Attend to next block
            if block_idx < num_blocks - 1:
                next_start = (block_idx + 1) * self.block_size
                next_end = next_start + self.block_size
                mask[start_pos:end_pos, next_start:next_end] = True
        
        # Random attention
        rand_block_indices = self._get_random_block_indices(num_blocks, device)
        
        for local_idx, block_idx in enumerate(range(1, num_blocks - 1)):
            start_pos   = block_idx * self.block_size
            end_pos     = start_pos + self.block_size
            
            # Get random blocks for this block
            random_blocks = rand_block_indices[local_idx]
            
            for rand_block in random_blocks:
                rand_start = rand_block * self.block_size
                rand_end = rand_start + self.block_size
                mask[start_pos:end_pos, rand_start:rand_end] = True
        
        return mask
    
    def forward(self, hidden_states, attention_mask=None, position_bias=None):

        B, L, D = hidden_states.shape
        
        # Verify sequence length
        if L % self.block_size != 0:
            raise ValueError(
                f"Sequence length ({L}) must be divisible by block_size ({self.block_size})"
            )
        
        # Compute Q, K, V
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)
        
        # Reshape: (B, L, D) -> (B, num_heads, L, head_dim)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Masking
        sparse_mask = self._create_sparse_attention_mask(L, hidden_states.device)
        sparse_mask = sparse_mask.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply sparse mask
        scores = scores.masked_fill(~sparse_mask, float('-inf'))
        
        # Apply ALiBi bias
        if self.alibi is not None:
            alibi_bias = self.alibi(L, L, hidden_states.device)
            # Apply only to non-masked positions
            scores = scores + alibi_bias.masked_fill(~sparse_mask, 0.0)
        
        # Apply position bias
        if position_bias is not None:
            scores = scores + position_bias
        
        # Apply padding mask
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape back: (B, num_heads, L, head_dim) -> (B, L, D)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.o(out)
        
        # None for position bias
        return out, None


######################
#### Feed Forward ####
######################


class FeedForward(nn.Module):
    """ Gated MLP based on GeGLU """
    
    def __init__(self, embed_dim, ff_dim, dropout=0.0, activation='gelu_new'):
        super().__init__()
        
        # Gated linear unit style
        self.wi_0       = nn.Linear(embed_dim, ff_dim, bias=False)  # Gate projection
        self.wi_1       = nn.Linear(embed_dim, ff_dim, bias=False)  # Up projection
        self.wo         = nn.Linear(ff_dim, embed_dim, bias=False)  # Down projection
        self.dropout    = nn.Dropout(dropout)
        
        if activation == 'gelu_new':
            self.act = nn.GELU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else:
            self.act = nn.GELU()
    
    def forward(self, x):

        # Gated activation: act(gate) * up
        hidden = self.act(self.wi_0(x)) * self.wi_1(x)
        hidden = self.dropout(hidden)
        output = self.wo(hidden)
        return output


class MoEFeedForward(nn.Module):
    """
    Mixture of Experts Feed-Forward Network
    
    Routes tokens to different expert networks based on learned gating.
    Uses top-k routing where each token is processed by k experts.
    """
    
    def __init__(
        self, 
        embed_dim,                  # Input/output dimension
        ff_dim,                     
        num_experts=8,              # Number of expert networks
        top_k=2,                    # Number of experts to route each token to
        dropout=0.0,                
        activation='gelu_new',
        load_balance=0.01           # Weight for load balancing auxiliary loss
    ):
        super().__init__()
        
        self.embed_dim      = embed_dim
        self.ff_dim         = ff_dim
        self.num_experts    = num_experts
        self.top_k          = top_k
        self.load_balance   = load_balance
        
        # Gating network - routes tokens to experts
        self.gate = nn.Linear(embed_dim, num_experts, bias=False)
        
        # Expert networks - each is a GeGLU-style FFN
        self.experts = nn.ModuleList([
            FeedForward(embed_dim, ff_dim, dropout, activation)
            for _ in range(num_experts)
        ])
        
        # For tracking expert usage (load balancing)
        self.register_buffer('expert_counts', torch.zeros(num_experts))
    
    def forward(self, x):
        """
        Args:
            x: (B, L, D) input tensor
        
        Returns:
            output: (B, L, D) processed tensor
            aux_loss: Load balancing auxiliary loss (scalar)
        """
        B, L, D = x.shape
        
        # Flatten batch and sequence for processing
        x_flat = x.view(-1, D)  # (B*L, D)
        
        # Compute gating scores
        gate_logits = self.gate(x_flat)  # (B*L, num_experts)
        gate_scores = F.softmax(gate_logits, dim=-1)
        
        # Select top-k experts
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        
        # Normalize top-k scores to sum to 1
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process through selected experts
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]  # (B*L,)
            expert_scores = top_k_scores[:, i:i+1]  # (B*L, 1)
            
            # Process each expert
            for expert_idx in range(self.num_experts):
                # Mask for tokens routed to this expert
                mask = (expert_indices == expert_idx)
                
                if mask.any():
                    # Get tokens for this expert
                    expert_input = x_flat[mask]
                    
                    # Process through expert
                    expert_output = self.experts[expert_idx](expert_input)
                    
                    # Weight by gating score and accumulate
                    output[mask] += expert_scores[mask] * expert_output
                    
                    # Track expert usage
                    if self.training:
                        self.expert_counts[expert_idx] += mask.sum().item()
        
        # Reshape back
        output = output.view(B, L, D)
        
        # Compute load balancing loss
        aux_loss = self._load_balancing_loss(gate_scores)
        
        return output, aux_loss
    
    def _load_balancing_loss(self, gate_scores):
        """
        Encourages balanced expert usage
        
        Args:
            gate_scores: (B*L, num_experts) gating probabilities
        
        Returns:
            loss: Scalar load balancing loss
        """
        # Average probability of routing to each expert
        expert_probs = gate_scores.mean(dim=0)  # (num_experts,)
        
        # Fraction of tokens routed to each expert (using top-k)
        top_k_mask = torch.zeros_like(gate_scores)
        _, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        top_k_mask.scatter_(1, top_k_indices, 1.0)
        expert_usage = top_k_mask.mean(dim=0)  # (num_experts,)
        
        # Loss: variance of (probs * usage) encourages balance
        # Ideally each expert has 1/num_experts probability and usage
        loss = self.num_experts * torch.sum(expert_probs * expert_usage)
        
        return self.load_balance * loss


################################
#### Relative Position Bias ####
################################


class RelativePositionBias(nn.Module):
    """
        Compute bucket indices for relative positions
        
        T5 uses a mix of:
            - Exact positions for small distances
            - Log-spaced bins for larger distances
    """
    
    def __init__(self, num_heads , num_buckets=32, max_distance=128, bidirectional=True):
        super().__init__()
        
        self.num_heads      = num_heads
        self.num_buckets    = num_buckets
        self.max_distance   = max_distance
        # Encoder is bidirectional
        self.bidirectional  = bidirectional
        
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
    
    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets, max_distance, bidirectional=True):

        relative_buckets = 0
        
        if bidirectional:
            num_buckets     //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, 
                torch.zeros_like(relative_position)
            )
        
        # Half buckets for exact positions
        max_exact   = num_buckets // 2
        is_small    = relative_position < max_exact
        
        # Half buckets for log-spaced bins
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        
        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )
        
        return relative_buckets
    
    def forward(self, query_length, key_length, device):

        context_position    = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position     = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position   = memory_position - context_position
        
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance
        )
        
        values = self.relative_attention_bias(relative_position_bucket)
        # (1, num_heads, query_len, key_len)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        
        return values


###############
#### ALiBi ####
###############


class ALiBi(nn.Module):
    """
    Attention with Linear Biases
    
    Reference: "Train Short, Test Long" (Press et al., 2021)
    """
    
    def __init__(self, num_heads, max_seq_len=8192):
        super().__init__()
        
        self.num_heads      = num_heads
        self.max_seq_len    = max_seq_len
        
        # Compute slopes for each head (geometric sequence)
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)
        
        # Precompute bias for max_seq_len
        alibi_bias = self._build_alibi_bias(max_seq_len, slopes)
        self.register_buffer('alibi_bias', alibi_bias)
    
    @staticmethod
    def _get_slopes(num_heads):

        def get_slopes_power_of_2(n):
            start = 2 ** (-2 ** -(math.log2(n) - 3))
            ratio = start
            return torch.tensor([start * (ratio ** i) for i in range(n)])
        
        # Handle non-power-of-2 num_heads
        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)
        else:
            # Closest power of 2
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            slopes = torch.cat([
                get_slopes_power_of_2(closest_power_of_2),
                get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:num_heads - closest_power_of_2]
            ])
            return slopes
    
    @staticmethod
    def _build_alibi_bias(seq_len, slopes):
        """ Build ALiBi bias matrix: -m * |i - j| """
        
        # Distance matrix |i - j|
        context_position    = torch.arange(seq_len)[:, None]
        memory_position     = torch.arange(seq_len)[None, :]
        relative_position   = torch.abs(memory_position - context_position)  # (seq_len, seq_len)
        
        # Apply head-specific slopes: -m * |i - j|
        slopes              = slopes[:, None, None]
        relative_position   = relative_position[None, :, :].float()
        
        alibi_bias = -slopes * relative_position    # (num_heads, seq_len, seq_len)
        alibi_bias = alibi_bias.unsqueeze(0)        # (1, num_heads, seq_len, seq_len)
        
        return alibi_bias
    
    def forward(self, query_length, key_length, device):

        # Use cached bias if within max_seq_len
        if query_length <= self.max_seq_len and key_length <= self.max_seq_len:
            bias = self.alibi_bias[:, :, :query_length, :key_length].to(device)
        else:
            # Recompute for longer sequences
            slopes = self.slopes.to(device)
            bias = self._build_alibi_bias(max(query_length, key_length), slopes)
            bias = bias[:, :, :query_length, :key_length].to(device)
        
        return bias


#################
#### Encoder ####
#################


class EncoderBlock(nn.Module):
    """
    Encoder block with pre-norm architecture
    
    Structure:
        - LayerNorm -> Self-Attention (standard or BigBird sparse) -> Residual
        - LayerNorm -> FeedForward -> Residual
    """
    
    def __init__(
        self, 
        embed_dim,
        num_heads, 
        ff_dim, 
        dropout=0.0, 
        attn_dropout=0.0, 
        has_relative_bias=True,
        use_alibi=False,
        use_bigbird_sparse=False,
        block_size=64,
        num_rand_blocks=3
    ):
        super().__init__()
        
        self.use_bigbird_sparse = use_bigbird_sparse
        
        # Attention
        if use_bigbird_sparse:
            self.self_attn = BigBirdSparseAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                block_size=block_size,
                num_rand_blocks=num_rand_blocks,
                dropout=attn_dropout,
                use_alibi=use_alibi
            )
        else:
            self.self_attn = Attention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=attn_dropout,
                is_decoder=False,
                is_cross_attention=False,
                has_relative_bias=has_relative_bias and not use_alibi,
                use_alibi=use_alibi
            )
        
        self.norm1      = LayerNorm(embed_dim)
        self.ff         = FeedForward(embed_dim, ff_dim, dropout)
        self.norm2      = LayerNorm(embed_dim)
        self.dropout    = nn.Dropout(dropout)
    
    def forward(self, hidden_states, attention_mask=None, position_bias=None):
        
        # Pre-norm
        normed = self.norm1(hidden_states)
        
        # Self-attention
        attn_output, position_bias = self.self_attn(
            normed,
            attention_mask=attention_mask,
            position_bias=position_bias
        )
        
        hidden_states = hidden_states + self.dropout(attn_output)
        
        # Pre-norm
        normed = self.norm2(hidden_states)
        
        # Feed forward
        ff_output       = self.ff(normed)
        hidden_states   = hidden_states + self.dropout(ff_output)
        
        return hidden_states, position_bias


class Encoder(nn.Module):
    
    def __init__(
        self, 
        num_layers, 
        embed_dim, 
        num_heads, 
        ff_dim, 
        dropout=0.0, 
        attn_dropout=0.0,
        use_alibi=False,
        use_bigbird_sparse=False,
        block_size=64,
        num_rand_blocks=3
    ):
        super().__init__()
        
        self.use_bigbird_sparse = use_bigbird_sparse
        self.use_alibi          = use_alibi
        
        # Only first layer computes position bias (not used with BigBird or ALiBi)
        # ALiBi is computed every layer (it's cheap - just a static bias)
        self.layers = nn.ModuleList([
            EncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                attn_dropout=attn_dropout,
                has_relative_bias=(i == 0 and not use_bigbird_sparse and not use_alibi),
                use_alibi=use_alibi,
                use_bigbird_sparse=use_bigbird_sparse,
                block_size=block_size,
                num_rand_blocks=num_rand_blocks
            )
            for i in range(num_layers)
        ])
        
        self.final_norm = LayerNorm(embed_dim)
        self.dropout    = nn.Dropout(dropout)
    
    def forward(self, hidden_states, attention_mask=None):

        position_bias = None
        
        for layer in self.layers:
            hidden_states, position_bias = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias if not self.use_bigbird_sparse else None
            )
        
        hidden_states = self.final_norm(hidden_states)
        # (B, L, D)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states


#################
#### Decoder ####
#################


class DecoderBlock(nn.Module):
    """
    Decoder block with pre-norm architecture
    
    Structure:
        - LayerNorm -> Causal Self-Attention -> Residual
        - LayerNorm -> Cross-Attention -> Residual
        - LayerNorm -> FeedForward/MoE -> Residual
    """
    
    def __init__(
        self, 
        embed_dim, 
        num_heads, 
        ff_dim, 
        dropout=0.0, 
        attn_dropout=0.0, 
        use_moe=False,
        num_experts=8,
        moe_top_k=2,
        moe_load_balance=0.01
    ):
        super().__init__()
        
        self.use_moe = use_moe
        
        # Self-attention
        self.self_attn = Attention(
            embed_dim           =embed_dim,
            num_heads           =num_heads,
            dropout             =attn_dropout,
            is_decoder          =True,
            is_cross_attention  =False,
            has_relative_bias   =False,
            use_alibi           =False
        )
        self.norm1 = LayerNorm(embed_dim)
        
        # Cross-attention to encoder
        self.cross_attn = Attention(
            embed_dim           =embed_dim,
            num_heads           =num_heads,
            dropout             =attn_dropout,
            is_decoder          =True,
            is_cross_attention  =True,
            has_relative_bias   =False,
            use_alibi           =False
        )
        self.norm2 = LayerNorm(embed_dim)
        
        # Feed-forward: MoE or standard
        if use_moe:
            self.ff = MoEFeedForward(
                embed_dim=embed_dim,
                ff_dim=ff_dim,
                num_experts=num_experts,
                top_k=moe_top_k,
                dropout=dropout,
                load_balance=moe_load_balance
            )
        else:
            self.ff = FeedForward(embed_dim, ff_dim, dropout)
        
        self.norm3      = LayerNorm(embed_dim)
        self.dropout    = nn.Dropout(dropout)
    
    def forward(self, hidden_states, encoder_hidden_states, attention_mask=None, encoder_attention_mask=None, position_bias=None):
        
        # Self-attention
        normed = self.norm1(hidden_states)
        attn_output, position_bias = self.self_attn(
            normed,
            attention_mask  =attention_mask,
            position_bias   =position_bias
        )
        hidden_states = hidden_states + self.dropout(attn_output)
        
        # Cross-attention
        normed = self.norm2(hidden_states)
        cross_output, _ = self.cross_attn(
            normed,
            key_value_states    =encoder_hidden_states,
            attention_mask      =encoder_attention_mask
        )
        hidden_states = hidden_states + self.dropout(cross_output)
        
        # Feed-forward
        normed = self.norm3(hidden_states)
        
        if self.use_moe:
            ff_output, moe_aux_loss = self.ff(normed)
            hidden_states           = hidden_states + self.dropout(ff_output)
            return hidden_states, position_bias, moe_aux_loss
        else:
            ff_output       = self.ff(normed)
            hidden_states   = hidden_states + self.dropout(ff_output)
            return hidden_states, position_bias, None


class Decoder(nn.Module):
    """Decoder stack with shared relative position bias and optional MoE"""
    
    def __init__(
        self, 
        num_layers, 
        embed_dim, 
        num_heads, 
        ff_dim, 
        dropout         =0.0, 
        attn_dropout    =0.0,
        use_moe         =True,
        num_experts     =8,
        moe_top_k       =2,
        moe_load_balance=0.01
    ):
        super().__init__()
        
        self.use_moe    = use_moe
        self.use_alibi  = use_alibi
        
        self.layers = nn.ModuleList([
            DecoderBlock(
                embed_dim           =embed_dim,
                num_heads           =num_heads,
                ff_dim              =ff_dim,
                dropout             =dropout,
                attn_dropout        =attn_dropout,
                has_relative_bias   =False,
                use_alibi           =False,
                use_moe             =use_moe,
                num_experts         =num_experts,
                moe_top_k           =moe_top_k,
                moe_load_balance    =moe_load_balance
            )
            for i in range(num_layers)
        ])
        
        self.final_norm = LayerNorm(embed_dim)
        self.dropout    = nn.Dropout(dropout)
    
    def forward(self, hidden_states, encoder_hidden_states, attention_mask=None, encoder_attention_mask=None):

        position_bias   = None
        total_moe_loss  = 0.0 if self.use_moe else None
        
        for layer in self.layers:
            result = layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                position_bias=position_bias
            )
            
            if self.use_moe:
                hidden_states, position_bias, moe_aux_loss = result
                if moe_aux_loss is not None:
                    total_moe_loss = total_moe_loss + moe_aux_loss
            else:
                hidden_states, position_bias, _ = result
        
        hidden_states = self.final_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states, total_moe_loss