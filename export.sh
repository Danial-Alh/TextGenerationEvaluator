#!/usr/bin/env bash
set -x
#python main.py real -d emnlp60-train -a export -m real dgsan mle newmle seqgan rankgan maligan -r last_iter last_iter nll last_iter bleu4 bleu4 bleu4 -k 0
python main.py real -d chpoem5-train -a export -m real dgsan mle newmle seqgan rankgan maligan -r last_iter last_iter nll last_iter bleu4 bleu4 bleu4 -k 0
#python main.py real -d wiki72-train -a export -m real dgsan mle newmle seqgan rankgan maligan -r last_iter last_iter nll last_iter bleu4 bleu4 bleu4 -k 0
#python main.py real -d threecorpus75-train -a export -m real dgsan mle newmle seqgan rankgan maligan -r last_iter last_iter nll last_iter bleu4 bleu4 bleu4 -k 0
python main.py real -d imdb30-train -a export -m real dgsan mle newmle seqgan rankgan maligan -r last_iter last_iter nll last_iter bleu4 bleu4 bleu4 -k 0
python main.py real -d coco60-train -a export -m real dgsan mle newmle seqgan rankgan maligan -r last_iter last_iter nll last_iter bleu4 bleu4 bleu4 -k 0
set +x
#python3 main.py real -d emnlp60-train -a gen -k 1 -m mle seqgan maligan rankgan -r last_iter bleu4 bleu4 bleu4
#python3 main.py real -d chpoem5-train -a gen -k 1 -m mle seqgan maligan rankgan -r last_iter bleu4 bleu4 bleu4
