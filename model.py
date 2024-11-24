import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(x):
    return x if isinstance(x, tuple) else (x,x)


class MLPBlock(nn.Module):
    def __init__(self, dim, mlp_dim, dropout_rate=0.2):
        super().__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        x = self.feedforward(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, dropout_rate=0.):
        super().__init__()
        self.heads = heads
        self.inner = head_dim * heads
        inner_dim = head_dim * heads

        project_out = not (heads == 1 and head_dim == dim)

        self.scale = head_dim ** (-1/2)

        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)

        self.attention = nn.Softmax()

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(p=dropout_rate),
        ) if project_out else nn.Identity()

    
    def forward(self, x):

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        score = torch.matmul(q, k.transpose(-1,-2)) * self.scale

        attention= self.attention(score)

        out = torch.matmul(attention, v)

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)
    

class TransformerBlock(nn.Module):

    def __init__(self, dim, heads, head_dim, dropout_rate=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mha = AttentionBlock(
            dim=dim, 
            heads=heads, 
            head_dim=head_dim,
            dropout_rate=dropout_rate,
        )

        mlp_dim = head_dim * 2

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(
            dim=dim,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
        )
    
    def forward(self, x):
        print(x.shape)
        x = self.mha(self.norm1(x)) + x
        x = self.mlp(self.norm2(x)) + x
        return x



class ViT(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, num_classes, dropout_rate=0., channels=3, pool='mean'):
        super().__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image size is not divisble by patch size'

        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        num_patches = (image_height // patch_height) * (image_width // patch_width)

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                p1 = patch_height,
                p2 = patch_width,
            ),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout_rate)

        self.transformer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.transformer_blocks.append(TransformerBlock(
                dim=dim,
                heads=heads,
                head_dim=patch_dim,
                dropout_rate=dropout_rate,
                ))

        self.pool = pool
        self.to_latent = nn.Identity()
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        return self.mlp_head(x)
        
