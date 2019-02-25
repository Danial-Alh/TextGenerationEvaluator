#!/usr/bin/env bash
set -x
./code_collector.sh phd
#./code_collector.sh v_emnlp
#./code_collector.sh under_emnlp
#python3 main.py real -d wiki72-train -a gen -k 2 -m mle seqgan maligan rankgan -r last_iter bleu4 bleu4 bleu4
#python3 main.py real -d coco60-train -a gen -k 1 2 -m mle seqgan maligan rankgan -r last_iter bleu4 bleu4 bleu4
#python3 main.py real -d imdb30-train -a gen -k 1 2 -m mle seqgan maligan rankgan -r last_iter bleu4 bleu4 bleu4
#python3 main.py real -d threecorpus75-train -a gen -k 1 2 -m mle seqgan maligan rankgan -r last_iter bleu4 bleu4 bleu4
#python3 main.py real -d emnlp60-train -a gen -k 1 2 -m mle seqgan maligan rankgan -r last_iter bleu4 bleu4 bleu4
python3 main.py real -d chpoem5-train -a gen -k 1 2 -m mle mle seqgan maligan rankgan -r last_iter nll bleu4 bleu4 bleu4
######################################################
#python3 main.py real -d wiki72-train -a gen -k 0 1 2 -m mle -r nll
#python3 main.py real -d coco60-train -a gen -k 0 1 2 -m mle -r nll
#python3 main.py real -d imdb30-train -a gen -k 0 1 2 -m mle -r nll
#python3 main.py real -d threecorpus75-train -a gen -k 0 1 2 -m mle -r nll
#python3 main.py real -d emnlp60-train -a gen -k 0 1 2 -m mle -r nll
#python3 main.py real -d chpoem5-train -a gen -k 1 2 -m mle -r nll
./code_collector.sh ipm_up
set +x
#python3 main.py real -d emnlp60-train -a gen -k 1 -m mle seqgan maligan rankgan -r last_iter bleu4 bleu4 bleu4
#python3 main.py real -d chpoem5-train -a gen -k 1 -m mle seqgan maligan rankgan -r last_iter bleu4 bleu4 bleu4
