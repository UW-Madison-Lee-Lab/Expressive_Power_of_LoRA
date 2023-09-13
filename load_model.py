import torch
from torch import nn

class LoRA(nn.Module):
    def __init__(self, width, rank, std = .25):
        super().__init__()
        
        # initialize the parameters
        self.lora_A = nn.Parameter(torch.randn(width, rank))
        self.lora_A.data.fill_(0.00)
        self.lora_A.requires_grad = True
        
        self.lora_B = nn.Parameter(torch.randn(width, rank))
        nn.init.normal_(self.lora_B, mean=0.0, std=std)
        self.lora_B.requires_grad = True
        
    def forward(self, x):
        return torch.matmul(x, self.lora_B @ self.lora_A.T)

class FNN(nn.Module):
    def __init__(
        self, 
        depth, 
        width, 
        rank, 
        std = .25,
        use_bias = True,
        apply_lora = True,
        activation = 'relu',
    ):
        super().__init__()
        self.depth = depth
        self.width = width  
        self.rank = rank
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'linear':
            self.activation = lambda x: x
        else:
            raise NotImplementedError(f'We only support relu and linear activation, and {activation} is not supported.')
        
        self.apply_lora = apply_lora
        if self.apply_lora: 
            self.loralist = nn.ModuleList([LoRA(width, rank, std) for _ in range(depth)])
        
        self.linearlist = nn.ModuleList([nn.Linear(width, width, bias=use_bias) for _ in range(depth)])
        
    def forward(self, x):
        for l in range(self.depth):
            linear_x = self.linearlist[l](x)
            lora_x = self.loralist[l](x) if self.apply_lora else 0
            x = linear_x + lora_x
            
            x = self.activation(x)         
        return x
        
class TFB(nn.Module):
    def __init__(self, embed_dim, n_head):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(embed_dim, n_head)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
    def forward(self, x):
        # multi-head attention
        attn_output, _ = self.attention(x, x, x)
        
        # feed-forward network
        ff_output = self.feed_forward(attn_output)
        
        return ff_output
    
class TFN(nn.Module):
    def __init__(self, embed_dim, n_head, depth):
        super().__init__()
        
        self.tfblist = nn.ModuleList([TFB(embed_dim, n_head) for _ in range(depth)])
        self.output_layer = nn.Linear(embed_dim, embed_dim, bias = False)
        
    def forward(self, x):
        for block in self.tfblist:
            x = block(x)
            
        x = self.output_layer(x)
        
        return x