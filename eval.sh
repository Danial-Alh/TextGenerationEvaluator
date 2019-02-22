#!/usr/bin/env bash
k=$1

evaluate () {
    set -x
    CUDA_VISIBLE_DEVICES=0 python3 main.py real -d $1-train -a eval -k $k -m mle seqgan maligan rankgan dgsan -r last_iter bleu4 bleu4 bleu4 last_iter
    set +x
    #CUDA_VISIBLE_DEVICES=0 python3 main.py real -d $data-train -a eval -m -k $1 mle dgsan -r last_iter
}

#if [ $k = "0" ]; then
#    evaluate "chpoem5"
#fi
#evaluate "coco60"
#evaluate "imdb30"
evaluate "threecorpus75"
evaluate "wiki72"
evaluate "emnlp60"