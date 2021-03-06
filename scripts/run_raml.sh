#!/bin/bash

train_src="data/train_news-commentary-v11.de-en.en"
train_tgt="data/train_news-commentary-v11.de-en.de"
dev_src="data/eval_news-commentary-v11.de-en.en"
dev_tgt="data/eval_news-commentary-v11.de-en.de"
test_src="data/test_news-commentary-v11.de-en.en"
test_tgt="data/test_news-commentary-v11.de-en.de"

#train_src="data/train_news-commentary-v11.de-en.de"
#train_tgt="data/train_news-commentary-v11.de-en.en"
#dev_src="data/eval_news-commentary-v11.de-en.de"
#dev_tgt="data/eval_news-commentary-v11.de-en.en"
#test_src="data/test_news-commentary-v11.de-en.de"
#test_tgt="data/test_news-commentary-v11.de-en.en"

job_name="news.raml.en-de.test"
#job_name="news.raml.de-en.test"
train_log="train."${job_name}".log"
model_name="model."${job_name}
job_file="scripts/train."${job_name}".sh"
decode_file=${job_name}".test.en"
temp="0.6"

#    --vocab data/news_vocab.de-en.bin \
#    --raml_sample_file data/samples.de-en.txt
python nmt.py \
    --mode raml_train \
    --vocab data/news_vocab.bin \
    --save_to models/${model_name} \
    --valid_niter 15400 \
    --valid_metric ppl \
    --beam_size 5 \
    --batch_size 10 \
    --sample_size 10 \
    --hidden_size 256 \
    --embed_size 256 \
    --uniform_init 0.1 \
    --dropout 0.2 \
    --clip_grad 5.0 \
    --lr_decay 0.5 \
    --temp ${temp} \
    --raml_sample_file data/samples.en-de.txt \
    --train_src ${train_src} \
    --train_tgt ${train_tgt} \
    --dev_src ${dev_src} \
    --dev_tgt ${dev_tgt} \
    --cuda

python nmt.py \
    --mode test \
    --load_model models/${model_name}.bin \
    --beam_size 5 \
    --decode_max_time_step 100 \
    --save_to_file decode/${decode_file} \
    --test_src ${test_src} \
    --test_tgt ${test_tgt} \
    --cuda

 echo "test result" >> logs/${train_log}
 perl multi-bleu.perl ${test_tgt} < decode/${decode_file} >> logs/${train_log}
