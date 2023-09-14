from torch import nn
from helper import *

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
        
    def forward_matrix(self, x):
        # x: (batch_size, width)
        # lora_A: (width, rank)
        # lora_B: (width, rank)
        # lora_B @ lora_A^T: (width, width)
        # x @ lora_B @ lora_A^T: (batch_size, width)
        return torch.matmul(x, self.lora_B @ self.lora_A.T)
    
    def forward_tensor(self, X):
        # x: (batch_size, width, seq_length)
        # lora_A: (width, rank)
        # lora_B: (width, rank)
        # lora_A @ lora_B^T: (width, width)
        # matmul(self.lora_A @ self.lora_B.T, X): (batch_size, width, seq_length)
        return matmul(self.lora_A @ self.lora_B.T, X)
    
    def forward(self, x):
        # identify the shape of x
        if x.dim() <= 2:
            return self.forward_matrix(x)
        else:
            return self.forward_tensor(x)

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
    
class MultiheadAttention(nn.Module):
    """
    This class implment the multi-head attention layer defined in the paper
    "The Expressive Power of Low-Rank Adaptation"
    """
    
    # comments: I have tried to directly use the multihead attention 
    #           layer in pytorch, but it seems that their implementation 
    #           is weird and does not support square weight matrix for 
    #           each head...
    
    def __init__(
        self, 
        embed_dim, 
        n_head,
        rank,
        std,
        apply_lora,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.n_head = n_head
        
        # initialize the weight matrices for all heads
        # Wq, Wk, Wv, Wo: (embed_dim, embed_dim)
        self.Wq, self.Wk, self.Wv, self.Wo = [], [], [], []
        for _ in range(n_head):
            self.Wq.append(nn.Parameter(torch.randn(embed_dim, embed_dim)))
            self.Wk.append(nn.Parameter(torch.randn(embed_dim, embed_dim)))
            self.Wv.append(nn.Parameter(torch.randn(embed_dim, embed_dim)))
            self.Wo.append(nn.Parameter(torch.randn(embed_dim, embed_dim)))
            
        self.apply_lora = apply_lora
        if self.apply_lora:
            # for each head, we have adapter for Wq, Wk, Wv, Wo
            self.loralist = nn.ModuleList([LoRA(embed_dim, rank, std) for _ in range(n_head*4)])
            
            
    def forward(self, q, k, v):
        """
        Args:
            q: (batch_size, embed_dim, seq_length)
            k: (batch_size, embed_dim, seq_length)
            v: (batch_size, embed_dim, seq_length)
            
        Returns:
            (batch_size, embed_dim, seq_length)
        """
        
        result = torch.zeros_like(q)
        
        # compute the output for each head
        for h in range(self.n_head):
            # compute the attention score for each head
            # attn_score: (batch_size, seq_length, seq_length)
            if self.apply_lora:
                Wq = self.Wq[h] + self.loralist[h*4].lora_A @ self.loralist[h*4].lora_B.T
                Wk = self.Wk[h] + self.loralist[h*4+1].lora_A @ self.loralist[h*4+1].lora_B.T
                Wv = self.Wv[h] + self.loralist[h*4+2].lora_A @ self.loralist[h*4+2].lora_B.T
                Wo = self.Wo[h] + self.loralist[h*4+3].lora_A @ self.loralist[h*4+3].lora_B.T
                
            else:
                Wq, Wk, Wv, Wo = self.Wq[h], self.Wk[h], self.Wv[h], self.Wo[h]
                
            attn_score = matmul(Wq, q).T @ matmul(Wk, k)
            
            # compute the attention weights for each head
            # softmax is applied column-wise
            # attn_weights: (batch_size, seq_length, seq_length)
            attn_weights = torch.softmax(attn_score, dim = 1)
            
            # compute the output for each head
            # attn_output: (batch_size, embed_dim, seq_length)
            attn_output = matmul(Wv, v) @ attn_weights
            
            # project the output and combine the results from all heads
            # result: (batch_size, embed_dim, seq_length)
            result += matmul(Wo, attn_output)
        
        # batch_size * embed_dim * seq_length
        return result
        
class TFB(nn.Module):
    def __init__(
        self, 
        embed_dim, 
        n_head, 
        rank, 
        std,
        apply_lora,
        is_last,
        batch_size, 
        seq_length,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.batch_size = batch_size
        self.seq_length = seq_length
        
        # initialize the multi-head attention and feed-forward network
        self.attention = MultiheadAttention(
            embed_dim, 
            n_head,
            rank,
            std,
            apply_lora,
        )
        self.feed_forward = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim, bias = True),
            nn.Linear(embed_dim, embed_dim, bias = True),
        ])
        
        self.apply_lora = apply_lora
        self.is_last = is_last
        
        if self.apply_lora and self.is_last:
            # we add lora adapter for the second feed-forward layer if it is the last block
            self.W2_lora = LoRA(embed_dim, rank, std)
        
    def forward(self, x):
        """_summary_

        Args:
            x: (batch_size, embed_dim, seq_length)

        Returns:
            f2_output: (batch_size, embed_dim, seq_length)
        """
        # multi-head attention: (batch_size, embed_dim, seq_length)
        attn_output = self.attention(x, x, x)
        
        # feed-forward network
        
        # apply the first feedforward layer
        # 1. reshape attn_output to (batch_size * seq_length, embed_dim)
        attn_output_reshaped = attn_output.permute(0, 2, 1).reshape(-1, self.embed_dim)
        # 2. apply the first feed-forward layer -> (batch_size * seq_length, embed_dim)
        f1_output_reshaped = self.feed_forward[0](attn_output_reshaped)
        # 3. permute it back to (batch_size, embed_dim, seq_length)
        f1_output = f1_output_reshaped.reshape(self.batch_size, self.seq_length, self.embed_dim).permute(0, 2, 1)
        # 4. apply relu
        f1_output = torch.relu(f1_output)
        
        # apply the second feedforward layer
        # 1. reshape f1_output to (batch_size * seq_length, embed_dim)
        f1_output_reshaped = f1_output.permute(0, 2, 1).reshape(-1, self.embed_dim)
        # 2. apply the second feed-forward layer -> (batch_size * seq_length, embed_dim)
        f2_output_reshaped = self.feed_forward[1](f1_output_reshaped)
        # 3. permute it back to (batch_size, embed_dim, seq_length)
        f2_output = f2_output_reshaped.reshape(self.batch_size, self.seq_length, self.embed_dim).permute(0, 2, 1)
        # 4. apply lora if this is the last block
        if self.apply_lora and self.is_last:
            # f2_output: (batch_size, embed_dim, seq_length)
            f2_output += self.W2_lora(f1_output)
        
        return f2_output
    
class TFN(nn.Module):
    def __init__(
        self, 
        embed_dim, 
        n_head, 
        depth,
        rank,
        std,
        apply_lora,
        batch_size,
        seq_length,
    ):
        super().__init__()
        
        self.tfblist = nn.ModuleList([TFB(
                embed_dim, 
                n_head,
                rank,
                std,
                apply_lora,
                is_last = False,
                batch_size = batch_size,
                seq_length = seq_length,
            ) for _ in range(depth-1)]
        )
        
        self.tfblist.append(TFB(
                embed_dim, 
                n_head,
                rank,
                std,
                apply_lora,
                is_last = True,
                batch_size = batch_size,
                seq_length = seq_length,
        ))
        
        self.output_layer = nn.Linear(embed_dim, embed_dim, bias = False)
        
        self.apply_lora = apply_lora
        if self.apply_lora: 
            # apply lora to the output layer
            self.output_layer_lora = LoRA(embed_dim, rank, std)
        
    def forward(self, x):
        """_summary_

        Args:
            x: (batch_size, embed_dim, seq_length)

        Returns:
            x: (batch_size, embed_dim, seq_length)
        """
        for block in self.tfblist:
            x = block(x)
            
        x = self.output_layer(x) 
        if self.apply_lora:
            x += self.output_layer_lora(x)
            
        return x