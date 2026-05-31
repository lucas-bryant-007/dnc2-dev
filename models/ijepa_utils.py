import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Block

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: [H*W]
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def apply_masks(x, masks):
    """
    x: [L, D] (The full positional embedding grid, where L is total patches)
    masks: [B, N] (Indices of the patches we want)
    Returns: [B, N, D]
    """
    # 1. Expand the grid to match batch size: [B, L, D]
    B = masks.shape[0]
    x = x.unsqueeze(0).expand(B, -1, -1)
    
    # 2. Gather the specific embeddings based on indices
    # We need to expand masks to [B, N, D] to gather along the last dim
    D = x.shape[-1]
    masks_expanded = masks.unsqueeze(-1).expand(-1, -1, D)
    
    # 3. Gather
    return x.gather(1, masks_expanded)

class IJEPAPredictorViT(nn.Module):
    def __init__(
        self,
        num_patches: int,       # Total patches in image (e.g. 196 for 224x224 /w patch16)
        embed_dim: int,         # Dimension of the Encoder (input)
        predictor_embed_dim: int = 384,
        depth: int = 2,
        num_heads: int = 12,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop: float = 0.1,
        attn_drop: float = 0.1,
    ):
        super().__init__()
        
        # 1. Project Encoder Dim -> Predictor Dim
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        
        # 2. Learnable Mask Token (Shared across all targets)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # 3. Fixed Sin-Cos Positional Embeddings (Not learnable)
        # We store the full grid (e.g., 196x384) in a buffer
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(num_patches, predictor_embed_dim),
            requires_grad=False
        )
        # Initialize the fixed grid
        grid_size = int(num_patches**.5)
        pos_embed = get_2d_sincos_pos_embed(predictor_embed_dim, grid_size, cls_token=False)
        self.predictor_pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        # 4. Transformer Blocks (Using timm Block)
        self.blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=drop,
                attn_drop=attn_drop,
                norm_layer=nn.LayerNorm,
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

    def forward(self, x, context_masks, target_masks):
        """
        x: [B, N_ctxt, D_enc]  (Encoder output for context)
        context_masks: [B, N_ctxt] (Indices of context patches)
        target_masks: [B, 4, N_targ] (Indices of target patches, 4 blocks per img)
        """
        B_img, N_ctxt, _ = x.shape
        num_targets_per_img = target_masks.shape[1] # Should be 4
        N_targ = target_masks.shape[2]

        # 0. Fix masks -- subtract 1 to account for CLS token
        context_masks = context_masks - 1
        target_masks = target_masks - 1

        # ----------------------------------------------------------
        # 1. Prepare Context (x)
        # ----------------------------------------------------------
        # Project to predictor dim
        x = self.predictor_embed(x) # [B, N_ctxt, D_pred]

        # Add Positional Embeddings to Context
        # We grab the embeddings for the specific context indices
        # breakpoint()
        pos_embed_context = apply_masks(self.predictor_pos_embed, context_masks)
        x += pos_embed_context

        # ----------------------------------------------------------
        # 2. Prepare Targets (Mask Tokens)
        # ----------------------------------------------------------
        # Flatten targets so we process all 4 blocks as a batch
        # target_masks: [B, 4, N_t] -> [B*4, N_t]
        target_masks_flat = target_masks.flatten(0, 1)
        B_total = target_masks_flat.shape[0] # B * 4

        # Create Mask Tokens: [B*4, N_t, D_pred]
        mask_tokens = self.mask_token.repeat(B_total, N_targ, 1)

        # Add Positional Embeddings to Mask Tokens
        # We grab the embeddings for the specific target indices
        pos_embed_targets = apply_masks(self.predictor_pos_embed, target_masks_flat)
        mask_tokens += pos_embed_targets

        # ----------------------------------------------------------
        # 3. Concatenate (Context + Targets)
        # ----------------------------------------------------------
        # We need to repeat the context to match the number of targets.
        # Current x is [B, N_ctxt, D]. We need [B*4, N_ctxt, D].
        # We use repeat_interleave so Image 1 context aligns with Image 1 targets.
        x_repeated = x.repeat_interleave(num_targets_per_img, dim=0)

        # Concatenate: [B*4, N_ctxt + N_targ, D_pred]
        combined_seq = torch.cat([x_repeated, mask_tokens], dim=1)

        # ----------------------------------------------------------
        # 4. Forward Pass
        # ----------------------------------------------------------
        for blk in self.blocks:
            combined_seq = blk(combined_seq)
        
        combined_seq = self.norm(combined_seq)

        # ----------------------------------------------------------
        # 5. Extract Predictions
        # ----------------------------------------------------------
        # We only want the predictions for the mask tokens (the targets)
        # Slice off the context tokens.
        predictions = combined_seq[:, N_ctxt:] 
        
        # Project back to encoder dim
        predictions = self.predictor_proj(predictions) # [B*4, N_targ, D_enc]

        # Reshape back to [B, 4, N_targ, D_enc] if desired, or keep flattened
        return predictions.view(B_img, num_targets_per_img, N_targ, -1)