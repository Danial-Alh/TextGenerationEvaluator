#!/usr/bin/env bash
export_comm () {
    set -x
    python main.py real -d $1-train -a export -m dgsan mle seqgan rankgan maligan -r last_iter nll bleu4 bleu4 bleu4 -k 0
#    python main.py real -d $1-train -a export -m real dgsan mle seqgan rankgan maligan -r last_iter last_iter nll bleu4 bleu4 bleu4 -k 0
    set +x
}
export_comm emnlp60
export_comm chpoem5
export_comm wiki72
export_comm threecorpus75
export_comm imdb30
export_comm coco60

#python main.py real -d emnlp60-train -a export -m real dgsan mle newmle seqgan rankgan maligan -r last_iter last_iter nll last_iter bleu4 bleu4 bleu4 -k 0
#python main.py real -d chpoem5-train -a export -m real dgsan mle newmle seqgan rankgan maligan -r last_iter last_iter nll last_iter bleu4 bleu4 bleu4 -k 0
#python main.py real -d wiki72-train -a export -m real dgsan mle newmle seqgan rankgan maligan -r last_iter last_iter nll last_iter bleu4 bleu4 bleu4 -k 0
#python main.py real -d threecorpus75-train -a export -m real dgsan mle newmle seqgan rankgan maligan -r last_iter last_iter nll last_iter bleu4 bleu4 bleu4 -k 0
#python main.py real -d imdb30-train -a export -m real dgsan mle newmle seqgan rankgan maligan -r last_iter last_iter nll last_iter bleu4 bleu4 bleu4 -k 0
#python main.py real -d coco60-train -a export -m real dgsan mle newmle seqgan rankgan maligan -r last_iter last_iter nll last_iter bleu4 bleu4 bleu4 -k 0
