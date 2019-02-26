#!/usr/bin/env bash
k=$1

evaluate () {
    set -x
#    CUDA_VISIBLE_DEVICES=0 python3 main.py real -d $1-train -a eval -k $k -m mle -r nll
#    CUDA_VISIBLE_DEVICES=0 python3 main.py real -d $1-train -a eval -k $k -m mle newmle real seqgan maligan rankgan dgsan -r nll last_iter last_iter bleu4 bleu4 bleu4 last_iter
    CUDA_VISIBLE_DEVICES=0 python3 main.py real -d $1-train -a eval -k $k -m dgsane1 dgsane5 -r last_iter last_iter
    #CUDA_VISIBLE_DEVICES=0 python3 main.py real -d $data-train -a eval -m -k $1 mle dgsan -r last_iter
    set +x
}

#evaluate "chpoem5"
#evaluate "coco60"
evaluate "imdb30"
#evaluate "threecorpus75"
#evaluate "wiki72"
#evaluate "emnlp60"