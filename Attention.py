import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_model=2, row_dim=0, col_dim=1):
        super().__init__()


        self.w_q = nn.Linear(in_features=d_model,out_features=d_model, bias=False)
        self.w_k = nn.Linear(in_features=d_model,out_features=d_model, bias=False)
        self.w_v = nn.Linear(in_features=d_model,out_features=d_model, bias=False)

        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self,token_encoding):
        k = self.w_k(token_encoding)
        q = self.w_q(token_encoding)
        v = self.w_v(token_encoding)

        sims = torch.matmul(q,k.transpose(dim0=self.row_dim,dim1=self.col_dim))
        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**.5)

        attention_percents = F.softmax(scaled_sims,dim= self.col_dim)

        attention_scores = torch.matmul(attention_percents,v )


        return attention_scores


class MaskedSelfAttention(nn.Module):

    def __init__(self, d_model=2, row_dim=0, col_dim=1):

        super().__init__()

        self.w_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.w_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.w_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self,token_encoding,mask=None):
        k = self.w_k(token_encoding)
        q = self.w_q(token_encoding)
        v = self.w_v(token_encoding)

        sims = torch.matmul(q,k.transpose(dim0=self.row_dim,dim1=self.col_dim))
        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)
        
        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)

        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores



class Attention(nn.Module):

    def __init__(self, d_model=2, row_dim=0,col_dim=1):
        super().__init__()

        self.w_q = nn.Linear(in_features=d_model,out_features=d_model, bias=False)
        self.w_k = nn.Linear(in_features=d_model,out_features=d_model, bias=False)
        self.w_v = nn.Linear(in_features=d_model,out_features=d_model, bias=False)

        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self,encodings_for_q, encodings_for_k, encodings_for_v, mask=None):
        q = self.w_q(encodings_for_q)
        k = self.w_k(encodings_for_k)
        v = self.w_v(encodings_for_v)

        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))

        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)
            
        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)

        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores


class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 d_model=2,  
                 row_dim=0, 
                 col_dim=1, 
                 num_heads=1):
        
        super().__init__()

        self.heads = nn.ModuleList([ Attention(d_model=2,
                      row_dim=0,
                      col_dim=1) for _ in range(num_heads)])
        
        self.col_dim =col_dim

    def forward(self,encodings_for_q, encodings_for_k, encodings_for_v):

        return [head(encodings_for_q, 
                               encodings_for_k,
                               encodings_for_v)  for head in self.heads]
    

