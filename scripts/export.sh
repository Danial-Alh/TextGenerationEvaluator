#!/usr/bin/env bash
set -x
python main.py real -d coco60-train -a export -m mle seqgan rankgan maligan real -r nll bleu4 bleu4 bleu4 last_iter -t 1
python main.py real -d emnlp60-train -a export -m mle seqgan rankgan maligan real -r nll bleu4 bleu4 bleu4 last_iter -t 1
python main.py real -d imdb30-train -a export -m mle seqgan rankgan maligan real -r nll bleu4 bleu4 bleu4 last_iter -t 1
python main.py oracle -d oracle75-train -a export -m mle seqgan rankgan maligan real -r nll_oracle nll_oracle nll_oracle nll_oracle last_iter -t 1
set +x
