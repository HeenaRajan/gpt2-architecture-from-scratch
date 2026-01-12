import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,context_length, dropout, num_heads=2, qkv_bias=False):
        super().__init__()
        assert(d_out % num_heads == 0), \
        "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(d_out, d_out)
        self.dropout_layer = torch.nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))


    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape

        queries = self.W_query(x)  #(batch_size, num_tokens, d_out)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)   #(batch_size, num_tokens, num_heads, head_dim)
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        queries = queries.transpose(1, 2)   #(batch_size, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attention_scores = queries @ keys.transpose(2, 3) #dot product for each head

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attention_scores.masked_fill_(mask_bool, -torch.inf)

        attention_weights = torch.softmax(attention_scores /  keys.shape[-1]**0.5, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)

        context_vectors = (attention_weights @ values).transpose(1,2) #(batch_size, num_tokens, num_heads, head_dim)
        
        # self.d_out = self.num_heads * self.head_dim
        context_vectors = context_vectors.contiguous().view(batch_size, num_tokens, self.d_out)

        context_vectors = self.out_proj(context_vectors) #optional projection

        return context_vectors