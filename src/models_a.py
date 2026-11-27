import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
  def __init__(self, vocab_size, emb_dim, hidden, num_layers = 1, dropout = 0.3, pad_idx = 0):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx = pad_idx)
    self.lstm = nn.LSTM(emb_dim, hidden, num_layers = num_layers, batch_first = True,
                        bidirectional = True, dropout = dropout if num_layers > 1 else 0.0)
    self.drop = nn.Dropout(dropout)

  def forward(self, src, src_lens):
    x = self.drop(self.emb(src))
    packed = pack_padded_sequence(x, src_lens.cpu(), batch_first = True, enforce_sorted = False)
    out, (h,c) = self.lstm(packed)
    out, _ = pad_packed_sequence(out, batch_first = True, total_length = src.size(1))
    return out, h, c

class BiLSTMAttnNMT(nn.Module):
  def __init__(self, src_vocab, tgt_vocab, emb_dim = 256, hidden = 256, enc_layers = 1, dropout = 0.3, pad_idx = 0):
    super().__init__()
    self.pad_idx = pad_idx
    self.encoder = Encoder(src_vocab, emb_dim, hidden, enc_layers, dropout, pad_idx)
    self.bridge_h = nn.Linear(2 * hidden, hidden)
    self.bridge_c = nn.Linear(2 * hidden, hidden)
    self.dec_emb = nn.Embedding(tgt_vocab, emb_dim, padding_idx = pad_idx)
    self.dec_lstm = nn.LSTM(emb_dim, hidden, batch_first = True)
    self.attn_proj = nn.Linear(2*hidden, hidden, bias = False)
    self.combine = nn.Linear(3* hidden, hidden)
    self.out = nn.Linear(hidden, tgt_vocab)
    self.drop = nn.Dropout(dropout)

  def encode(self, src, src_lens):
    enc_out, h, c = self.encoder(src, src_lens)
    mask = src != self.pad_idx
    h0 = torch.tanh(self.bridge_h(torch.cat([h[-2], h[-1]], dim = -1))).unsqueeze(0)
    c0 = torch.tanh(self.bridge_c(torch.cat([c[-2], c[-1]], dim = -1))).unsqueeze(0)
    return enc_out, mask, (h0, c0)

  def attend(self, dec_out, enc_out, mask):
    keys = self.attn_proj(enc_out)
    scores = torch.bmm(dec_out, keys.transpose(1,2))
    scores = scores.masked_fill(~mask.unsqueeze(1), torch.finfo(scores.dtype).min)
    alpha = F.softmax(scores, dim = -1)
    context = torch.bmm(alpha, enc_out)
    return context, alpha

  def decode(self, tgt_in, state, enc_out, mask):
    x = self.drop(self.dec_emb(tgt_in))
    dec_out, state = self.dec_lstm(x, state)
    context, alpha = self.attend(dec_out, enc_out, mask)
    comb = torch.tanh(self.combine(torch.cat([context, dec_out], dim = -1)))
    logits = self.out(self.drop(comb))
    return logits, state, alpha

  def forward(self, src, src_lens, tgt_in):
    enc_out, mask, state = self.encode(src, src_lens)
    logits, _, alpha = self.decode(tgt_in, state, enc_out, mask)
    return logits, alpha
    
      
