import argparse
import os
from util import read_corpus, write_corpus


def make_data(src, tgt, src_file, tgt_file, train_ratio=0.5, eval_ratio=0.1):
  train_src = src[:int(len(src) * train_ratio)]
  eval_src = src[int(len(src) * train_ratio):int(len(src) * (train_ratio + eval_ratio))]
  test_src = src[int(len(src) * (train_ratio + eval_ratio)):]
  train_tgt = tgt[:int(len(tgt) * train_ratio)]
  eval_tgt = tgt[int(len(tgt) * train_ratio):int(len(tgt) * (train_ratio + eval_ratio))]
  test_tgt = tgt[int(len(tgt) * (train_ratio + eval_ratio)):]
  write_corpus(os.path.join("data", "train_" + src_file), train_src)
  write_corpus(os.path.join("data", "eval_" + src_file), eval_src)
  write_corpus(os.path.join("data", "test_" + src_file), test_src)
  write_corpus(os.path.join("data", "train_" + tgt_file), train_tgt)
  write_corpus(os.path.join("data", "eval_" + tgt_file), eval_tgt)
  write_corpus(os.path.join("data", "test_" + tgt_file), test_tgt)
  return


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_src', type=str, default='data/news-commentary-v11.de-en.en',
                      help='file of source sentences')
  parser.add_argument('--train_tgt', type=str, default='data/news-commentary-v11.de-en.de',
                      help='file of target sentences')
  parser.add_argument('--src_vocab_size', default=20000, type=int, help='source vocabulary size')
  parser.add_argument('--tgt_vocab_size', default=20000, type=int, help='target vocabulary size')
  
  args = parser.parse_args()
  
  print('read in source sentences: %s' % args.train_src)
  print('read in target sentences: %s' % args.train_tgt)
  
  src_sents = read_corpus(args.train_src, source='src', generate=True)[:args.src_vocab_size]
  tgt_sents = read_corpus(args.train_tgt, source='tgt', generate=True)[:args.tgt_vocab_size]
  
  assert len(src_sents) == len(tgt_sents)
  make_data(src_sents, tgt_sents, args.train_src.split("/")[-1], args.train_tgt.split("/")[-1])
