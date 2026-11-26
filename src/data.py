import re, random, unicodedata
from collections import Counter
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

PAD, UNK, SOS, EOS = "<pad>", "<unk>", "<sos>", "<eos>"
SPECIALS = [PAD, UNK, SOS, EOS]
PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3

def clean_text(text):
  #En mínusculas, sin acentos, sin puntuación. Es la misma función para entrenar y para BLEU
  text = unicodedata.normalize("NFD", text.lower())
  text = "".join(c for c in text if unicodedata.category(c) != "Mn")
  text = re.sub(r"[^\w\s]", " ", text)
  return re.sub(r"\s+", " ", text).strip()

def build_vocab(sentences, min_freq = 2, max_size = 20000):
  ## Construir solo oraciones a partir de train_pairs
  counter = Counter(w for s in sentences for w in clean_text(s).split())
  word2idx = {tok:i for i, tok in enumerate(SPECIALS)}
  for w, c in counter.most_common():
    if c < min_freq or len(word2idx) >= max_size:
      break
    word2idx[w] = len(word2idx)
  return word2idx

def encode(sentence, word2idx, add_sos_eos = False):
  ids = [word2idx.get(w, UNK_IDX) for w in clean_text(sentence).split()]
  return [SOS_IDX] + ids + [EOS_IDX] if add_sos_eos else ids

class NMTDataset(Dataset):
  def __init__(self, pairs, eng_word2idx, spa_word2idx):
    self.src = [torch.tensor(encode(en, eng_word2idx)) for en, _ in pairs]
    self.tgt = [torch.tensor(encode(es, spa_word2idx, add_sos_eos = True)) for _, es in pairs]

  def __len__(self):
    return len(self.src)

  def __getitem__(self, i):
    return self.src[i], self.tgt[i]

def collate_fn(batch):
  src, tgt = zip(*batch)
  src_lens = torch.tensor([len(s) for s in src])
  src = pad_sequence(src, batch_first = True, padding_value = PAD_IDX)
  tgt = pad_sequence(tgt, batch_first = True, padding_value = PAD_IDX)
  return src, src_lens, tgt

class BucketSampler:
  ## Agrupa oraciones de longitud similar para menos relleno de padding
  def __init__(self, lengths, batch_size, shuffle = True, chunk = 50):
    self.lengths, self.bs, self.shuffle, self.chunk = lengths, batch_size, shuffle, chunk

  def __iter__(self):
    idx = list(range(len(self.lengths)))
    if self.shuffle:
      random.shuffle(idx)
    size = self.bs * self.chunk
    batches = []
    for i in range(0, len(idx), size):
      c = sorted(idx[i:i + size], key = lambda j: self.lengths[j])
      batches += [c[k:k + self.bs] for k in range(0, len(c), self.bs)]
    if self.shuffle:
      random.shuffle(batches)
    yield from batches
  
  def __len__(self):
    return (len(self.lengths) + self.bs - 1) // self.bs
