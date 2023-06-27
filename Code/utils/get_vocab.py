from torchtext.vocab import build_vocab_from_iterator

def get_vocab(data, tokenizer):
  
  def yield_token(batch):
    for text, _ in batch:
      yield tokenizer(text)

  # Build vocab
  vocab = build_vocab_from_iterator(yield_token(data), specials=['<unk>', '<pad>'])
  vocab.set_default_index(vocab['<unk>'])

  return vocab
