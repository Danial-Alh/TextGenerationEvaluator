#!/usr/bin/env bash
set -x
python3 main.py real -d wiki72-train -a gen -k 1 -m mle seqgan maligan rankgan -r last_iter bleu4 bleu4 bleu4
python3 main.py real -d coco60-train -a gen -k 1 2 -m mle seqgan maligan rankgan -r last_iter bleu4 bleu4 bleu4
python3 main.py real -d imdb30-train -a gen -k 1 2 -m mle seqgan maligan rankgan -r last_iter bleu4 bleu4 bleu4
python3 main.py real -d threecorpus75-train -a gen -k 1 2 -m mle seqgan maligan rankgan -r last_iter bleu4 bleu4 bleu4
./code_collector.sh ipm
set +x
#python3 main.py real -d emnlp60-train -a gen -k 1 -m mle seqgan maligan rankgan -r last_iter bleu4 bleu4 bleu4
#python3 main.py real -d chpoem5-train -a gen -k 1 -m mle seqgan maligan rankgan -r last_iter bleu4 bleu4 bleu4