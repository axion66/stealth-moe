import torch
import torch.nn as nn
import torch.nn.functional as F
from basicts.utils import data_transformation_4_xformer
import numpy as np
import math
from math import sqrt


# ========================= MASKING CLASSES =========================
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


# ========================= EMBEDDING CLASSES =========================
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


# ========================= ATTENTION CLASSES =========================
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


# ========================= MIXTURE OF EXPERTS CLASSES =========================

class MoE(nn.Module):
    """
    Mixture of Experts layer with top-k gating
    """
    def __init__(self, input_dim, output_dim, num_experts, k, expert_dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.expert_dropout = expert_dropout

        # Expert networks - each expert is a small feedforward network
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(expert_dropout),
                nn.Linear(output_dim, output_dim)
            ) for _ in range(num_experts)
        ])

        # Gating network
        self.gate = nn.Linear(input_dim, num_experts)
        self.noise_generator = nn.Linear(input_dim, num_experts)
        
        # Load balancing
        self.register_buffer("expert_counts", torch.zeros(num_experts))
        self.load_balance_loss_weight = 0.01

    def forward(self, x, training=True):
        '''
            Args:
            x: (batch, seq, input_dim)
        
            Return:
            output: (batch, seq, output_dim)
            aux_loss: auxiliary loss for load balancing
        '''
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)  # [B*S, D]

        # Gate computation with noise for training
        gate_logits = self.gate(x_flat)  # [B*S, num_experts]
        
        if training and self.training:
            # Add noise for better exploration during training
            noise = torch.randn_like(gate_logits) * 0.1
            gate_logits = gate_logits + noise

        gate_scores = F.softmax(gate_logits, dim=-1)

        # Top-k selection
        topk_scores, topk_indices = torch.topk(gate_scores, self.k, dim=-1) # [B*S, k]
        topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-8)

        # Prepare output tensor
        output = torch.zeros(x_flat.shape[0], self.experts[0][0].out_features, 
                           device=x.device, dtype=x.dtype) # [B*S, output_dim]
        
        # Load balancing: track expert usage
        if self.training:
            for i in range(self.num_experts):
                count = (topk_indices == i).sum().float()
                self.expert_counts[i] = 0.9 * self.expert_counts[i] + 0.1 * count
        
        # Route to experts
        for expert_id in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (topk_indices == expert_id)
            
            if expert_mask.any():
                # Get indices of tokens for this expert
                token_indices, expert_indices = torch.where(expert_mask)
                expert_inputs = x_flat[token_indices]
                expert_weights = topk_scores[token_indices, expert_indices].unsqueeze(-1)
                
                # Forward through expert
                expert_outputs = self.experts[expert_id](expert_inputs) * expert_weights
                
                # Add to output
                output.index_add_(0, token_indices, expert_outputs)
        
        # Load balancing auxiliary loss
        aux_loss = self._compute_load_balance_loss(gate_scores)
        
        return output.view(batch_size, seq_len, -1), aux_loss
    
    def _compute_load_balance_loss(self, gate_scores):
        """Compute load balancing loss to encourage even expert usage"""
        if not self.training:
            return torch.tensor(0.0, device=gate_scores.device)
            
        # Compute load balancing loss
        expert_utilization = gate_scores.mean(dim=0)  # Average gate scores per expert
        expert_importance = (gate_scores > 0).float().mean(dim=0)  # Fraction of tokens routed to each expert
        
        load_loss = self.num_experts * torch.sum(expert_utilization * expert_importance)
        return self.load_balance_loss_weight * load_loss


class MoEFeedForward(nn.Module):
    """
    MoE-enhanced Feed Forward Network to replace traditional FFN in transformer layers
    """
    def __init__(self, d_model, d_ff, num_experts, k=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # MoE layer
        self.moe = MoE(d_model, d_ff, num_experts, k, expert_dropout=dropout)
        
        # Output projection
        self.output_proj = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            output: [batch, seq_len, d_model]
            aux_loss: auxiliary loss for load balancing
        """
        moe_output, aux_loss = self.moe(x, training=self.training)
        output = self.dropout(self.output_proj(moe_output))
        return output, aux_loss


class NSMoEEncoderLayer(nn.Module):
    """
    NS-MoE Enhanced Encoder Layer with MoE instead of traditional FFN
    """
    def __init__(self, attention, d_model, d_ff=None, num_experts=8, k=2, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        self.attention = attention
        self.moe_ffn = MoEFeedForward(d_model, d_ff, num_experts, k, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # Multi-head attention
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        # MoE Feed Forward
        moe_output, aux_loss = self.moe_ffn(x)
        x = x + moe_output
        x = self.norm2(x)

        return x, attn, aux_loss


class NSMoEDecoderLayer(nn.Module):
    """
    NS-MoE Enhanced Decoder Layer with MoE for both self-attention and cross-attention
    """
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, num_experts=8, k=2, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.moe_ffn = MoEFeedForward(d_model, d_ff, num_experts, k, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # Decoder self-attention
        new_x, self_attn = self.self_attention(x, x, x, attn_mask=x_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        # Decoder-encoder cross-attention
        new_x, cross_attn = self.cross_attention(x, cross, cross, attn_mask=cross_mask)
        x = x + self.dropout(new_x)
        x = self.norm2(x)

        # MoE Feed Forward
        moe_output, aux_loss = self.moe_ffn(x)
        x = x + moe_output
        x = self.norm3(x)

        return x, self_attn, cross_attn, aux_loss


class NSMoEEncoder(nn.Module):
    """
    NS-MoE Enhanced Encoder with multiple MoE layers
    """
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        total_aux_loss = 0.0
        
        for layer in self.layers:
            x, attn, aux_loss = layer(x, attn_mask=attn_mask)
            attns.append(attn)
            total_aux_loss += aux_loss

        if self.norm is not None:
            x = self.norm(x)

        return x, attns, total_aux_loss


class NSMoEDecoder(nn.Module):
    """
    NS-MoE Enhanced Decoder with multiple MoE layers and cross-attention
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        self_attns = []
        cross_attns = []
        total_aux_loss = 0.0
        
        for layer in self.layers:
            x, self_attn, cross_attn, aux_loss = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            self_attns.append(self_attn)
            cross_attns.append(cross_attn)
            total_aux_loss += aux_loss

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x, self_attns, cross_attns, total_aux_loss




class NSMOE(nn.Module):
    """
    Neural Scaling Mixture of Experts for Time Series Forecasting
    
    Encoder-Decoder architecture with traditional (non-inverted) transformer approach 
    where time steps are tokens, enhanced with Mixture of Experts in both encoder 
    and decoder feed-forward layers for better scaling and specialization.
    
    Key Features:
    - Traditional transformer approach (time steps as tokens)
    - Encoder-Decoder architecture with autoregressive generation
    - MoE-enhanced encoder and decoder layers
    - Load balancing for expert utilization
    - Auxiliary loss for training stability
    """

    def __init__(self, **model_args):
        super().__init__()
        
        # Model parameters
        self.pred_len = model_args['pred_len']
        self.seq_len = model_args['seq_len']
        self.label_len = model_args.get('label_len', model_args['seq_len'] // 2)
        self.output_attention = model_args.get('output_attention', False)
        self.enc_in = model_args['enc_in']
        self.dec_in = model_args['dec_in']
        self.c_out = model_args['c_out']
        
        # Transformer parameters
        self.d_model = model_args['d_model']
        self.n_heads = model_args['n_heads']
        self.d_ff = model_args['d_ff']
        self.e_layers = model_args['e_layers']
        self.d_layers = model_args.get('d_layers', 1)
        self.dropout = model_args.get('dropout', 0.1)
        self.activation = model_args.get('activation', 'relu')
        self.embed = model_args.get('embed', 'timeF')
        self.freq = model_args.get('freq', 'h')
        
        # MoE specific parameters
        self.num_experts = model_args.get('num_experts', 8)
        self.k = model_args.get('top_k_experts', 2)  # Number of experts to route to
        self.use_norm = model_args.get('use_norm', True)
        
        # Embeddings - Traditional approach: embed variables as features, time as sequence
        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, self.embed, self.freq, self.dropout)
        self.dec_embedding = DataEmbedding(self.dec_in, self.d_model, self.embed, self.freq, self.dropout)

        # NS-MoE Enhanced Encoder
        self.encoder = NSMoEEncoder(
            [
                NSMoEEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=self.dropout,
                                      output_attention=self.output_attention), 
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    num_experts=self.num_experts,
                    k=self.k,
                    dropout=self.dropout,
                    activation=self.activation
                ) for _ in range(self.e_layers)
            ],
            norm_layer=nn.LayerNorm(self.d_model)
        )
        
        # NS-MoE Enhanced Decoder
        self.decoder = NSMoEDecoder(
            [
                NSMoEDecoderLayer(
                    AttentionLayer(
                        FullAttention(True, attention_dropout=self.dropout,
                                      output_attention=False), 
                        self.d_model, self.n_heads),
                    AttentionLayer(
                        FullAttention(False, attention_dropout=self.dropout,
                                      output_attention=False), 
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    num_experts=self.num_experts,
                    k=self.k,
                    dropout=self.dropout,
                    activation=self.activation
                ) for _ in range(self.d_layers)
            ],
            norm_layer=nn.LayerNorm(self.d_model),
            projection=nn.Linear(self.d_model, self.c_out, bias=True)
        )
        
        # Auxiliary loss weight for MoE load balancing
        self.aux_loss_weight = model_args.get('aux_loss_weight', 0.01)

    def forward_nsmoe(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor,
                      x_mark_dec: torch.Tensor) -> tuple:
        """
        Forward pass through NS-MoE Encoder-Decoder architecture

        Args:
            x_enc: Input encoder data [B, L, N]
            x_mark_enc: Encoder time features [B, L, C-1]
            x_dec: Decoder data [B, label_len + pred_len, N] 
            x_mark_dec: Decoder time features [B, label_len + pred_len, C-1]

        Returns:
            prediction: [B, pred_len, N]
            aux_loss: Auxiliary loss for MoE load balancing
        """
        B, L, N = x_enc.shape
        B_dec, L_dec, N_dec = x_dec.shape

        if self.use_norm:
            # Normalization from Non-stationary Transformer - normalize across time dimension
            means = x_enc.mean(1, keepdim=True).detach()  # B, 1, N
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)  # B, 1, N
            x_enc /= stdev

        # Encoder: B L N -> B L E (Traditional Transformer approach)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # B L E
        
        # NS-MoE Enhanced Encoding: B L E -> B L E
        # Time steps are treated as tokens, processed by MoE-enhanced encoder layers
        enc_out, enc_attns, enc_aux_loss = self.encoder(enc_out, attn_mask=None)

        # Decoder: B L_dec N -> B L_dec E
        dec_out = self.dec_embedding(x_dec, x_mark_dec)  # B L_dec E
        
        # NS-MoE Enhanced Decoding: B L_dec E -> B L_dec N
        # Autoregressive generation with cross-attention to encoder
        dec_out, dec_self_attns, dec_cross_attns, dec_aux_loss = self.decoder(
            dec_out, enc_out, x_mask=None, cross_mask=None)

        # Extract prediction part (remove label_len tokens)
        prediction = dec_out[:, -self.pred_len:, :]  # B, pred_len, N

        if self.use_norm:
            # De-Normalization
            prediction = prediction * stdev.repeat(1, self.pred_len, 1)
            prediction = prediction + means.repeat(1, self.pred_len, 1)

        # Total auxiliary loss from both encoder and decoder
        total_aux_loss = enc_aux_loss + dec_aux_loss

        return prediction, total_aux_loss

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """
        Main forward pass for training/inference

        Args:
            history_data (Tensor): Input data with shape: [B, L1, N, C]
            future_data (Tensor): Future data with shape: [B, L2, N, C]
            batch_seen (int): Current batch number
            epoch (int): Current epoch
            train (bool): Training mode flag

        Returns:
            torch.Tensor: outputs with shape [B, L2, N, 1]
        """
        # Transform data for transformer processing
        x_enc, x_mark_enc, x_dec, x_mark_dec = data_transformation_4_xformer(
            history_data=history_data,
                                                                             future_data=future_data,
            start_token_len=self.label_len  # Use label_len for decoder start tokens
        )
        
        # Forward through NS-MoE
        prediction, aux_loss = self.forward_nsmoe(
            x_enc=x_enc, x_mark_enc=x_mark_enc, 
            x_dec=x_dec, x_mark_dec=x_mark_dec
        )
        
        # Store auxiliary loss for backward pass
        if train:
            self._aux_loss = aux_loss * self.aux_loss_weight
        else:
            self._aux_loss = torch.tensor(0.0, device=prediction.device)
        
        return prediction.unsqueeze(-1)  # B, L2, N, 1
    
    def get_aux_loss(self):
        """Get auxiliary loss for MoE load balancing"""
        return getattr(self, '_aux_loss', torch.tensor(0.0))


class MSNSMOE(nn.Module):
    """
    Multi-Scale Neural Scaling Mixture of Experts for Time Series Forecasting
    
    Processes input sequences at multiple resolutions (1, 2, 4) and combines predictions:
    - Scale 1: Full sequence (no split) - 1 NSMOE model
    - Scale 2: Split sequence into 2 parts - 2 NSMOE models, then combine
    - Scale 4: Split sequence into 4 parts - 4 NSMOE models, then combine
    
    Final prediction is the ensemble mean of all 3 full-length predictions.
    
    Key Features:
    - Multi-resolution temporal modeling
    - Ensemble of different temporal scales
    - Enhanced pattern capture from coarse to fine-grained
    """

    def __init__(self, **model_args):
        super().__init__()
        
        # Store model arguments for creating NSMOE models
        self.model_args = model_args
        
        # Model parameters
        self.pred_len = model_args['pred_len']
        self.seq_len = model_args['seq_len']
        self.label_len = model_args.get('label_len', model_args['seq_len'] // 2)
        self.enc_in = model_args['enc_in']
        self.dec_in = model_args['dec_in']
        self.c_out = model_args['c_out']
        
        # Multi-scale configuration
        self.scales = [1, 2, 4]  # Resolution scales
        self.use_norm = model_args.get('use_norm', True)
        
        # Create NSMOE models for each scale and each split
        self.scale_models = nn.ModuleDict()
        
        # Scale 1: Full sequence (1 model)
        self.scale_models['scale_1'] = nn.ModuleList([
            NSMOE(**self._get_scale_args(1))
        ])
        
        # Scale 2: Split into 2 parts (2 models)
        self.scale_models['scale_2'] = nn.ModuleList([
            NSMOE(**self._get_scale_args(2)) for _ in range(2)
        ])
        
        # Scale 4: Split into 4 parts (4 models)  
        self.scale_models['scale_4'] = nn.ModuleList([
            NSMOE(**self._get_scale_args(4)) for _ in range(4)
        ])
        
        # Auxiliary loss weight
        self.aux_loss_weight = model_args.get('aux_loss_weight', 0.01)
    
    def _get_scale_args(self, scale: int) -> dict:
        """
        Get model arguments adjusted for specific scale
        
        Args:
            scale: Resolution scale (1, 2, or 4)
            
        Returns:
            dict: Adjusted model arguments
        """
        args = self.model_args.copy()
        
        # Adjust sequence length for the scale (ensure divisibility)
        args['seq_len'] = max(1, self.seq_len // scale)
        args['label_len'] = max(1, self.label_len // scale)
        
        # Keep pred_len unchanged for all scales - we want to predict the same future length
        args['pred_len'] = self.pred_len
        
        return args
    
    def _split_sequence(self, x: torch.Tensor, scale: int) -> list:
        """
        Split input sequence into chunks based on scale
        
        Args:
            x: Input tensor [B, L, N, C] or [B, L, N]
            scale: Number of chunks to split into
            
        Returns:
            list: List of tensor chunks
        """
        B, L = x.shape[:2]
        chunk_size = L // scale
        
        chunks = []
        for i in range(scale):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunks.append(x[:, start_idx:end_idx])
        
        return chunks
    
    
    def forward_multi_scale(self, history_data: torch.Tensor, future_data: torch.Tensor, 
                           batch_seen: int, epoch: int, train: bool) -> tuple:
        """
        Forward pass through multi-scale architecture
        
        Args:
            history_data: Historical data [B, L1, N, C]
            future_data: Future data [B, L2, N, C]
            batch_seen: Batch number
            epoch: Current epoch
            train: Training mode flag
            
        Returns:
            tuple: (ensemble_prediction, total_aux_loss)
        """
        B, L1, N, C = history_data.shape
        B, L2, N_f, C_f = future_data.shape
        
        scale_predictions = []
        total_aux_loss = torch.tensor(0.0, device=history_data.device)
        
        # Process each scale
        for scale in self.scales:
            scale_name = f'scale_{scale}'
            models = self.scale_models[scale_name]
            
            if scale == 1:
                # Scale 1: Full sequence processing
                pred = models[0](history_data, future_data, batch_seen, epoch, train)
                scale_predictions.append(pred)
                
                # Add auxiliary loss
                if hasattr(models[0], 'get_aux_loss'):
                    total_aux_loss += models[0].get_aux_loss()
                    
            else:
                # Scale 2 or 4: Split history sequence processing  
                # Split only history data into chunks, keep future data full length
                history_chunks = self._split_sequence(history_data, scale)
                
                chunk_predictions = []
                for i, hist_chunk in enumerate(history_chunks):
                    # Each model processes its history chunk but predicts the full future
                    pred_chunk = models[i](hist_chunk, future_data, batch_seen, epoch, train)
                    chunk_predictions.append(pred_chunk)
                    
                    # Add auxiliary loss
                    if hasattr(models[i], 'get_aux_loss'):
                        total_aux_loss += models[i].get_aux_loss()
                
                # Average predictions from all chunks (ensemble across temporal chunks)
                combined_pred = torch.stack(chunk_predictions, dim=0).mean(dim=0)
                scale_predictions.append(combined_pred)
        
        # Ensemble: Average predictions from all scales
        ensemble_prediction = torch.stack(scale_predictions, dim=0).mean(dim=0)
        
        return ensemble_prediction, total_aux_loss
    
    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, 
               batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """
        Main forward pass for training/inference
        
        Args:
            history_data: Historical data [B, L1, N, C]
            future_data: Future data [B, L2, N, C]
            batch_seen: Batch number
            epoch: Current epoch
            train: Training mode flag
            
        Returns:
            torch.Tensor: Ensemble prediction [B, L2, N, C]
        """
        prediction, aux_loss = self.forward_multi_scale(
            history_data, future_data, batch_seen, epoch, train
        )
        
        # Store auxiliary loss for potential use during training
        if train and aux_loss.requires_grad:
            self._aux_loss = aux_loss * self.aux_loss_weight
        else:
            self._aux_loss = torch.tensor(0.0, device=prediction.device)
        
        return prediction.unsqueeze(-1)  # B, L2, N, 1
    
    def get_aux_loss(self):
        """Get auxiliary loss for MoE load balancing across all scales"""
        return getattr(self, '_aux_loss', torch.tensor(0.0))