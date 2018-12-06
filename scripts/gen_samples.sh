#!/bin/sh

train_src="data/train_news-commentary-v11.de-en.en"
train_tgt="data/train_news-commentary-v11.de-en.de"

python process_samples.py \
    --mode sample_ngram \
    --vocab data/iwslt.vocab.bin \
    --src ${train_src} \
    --tgt ${train_tgt} \
    --output /tmp/samples.txt