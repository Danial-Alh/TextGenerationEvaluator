#!/usr/bin/env bash
set -x
#./code_collector.sh phd
#./code_collector.sh v_emnlp
#./code_collector.sh under_emnlp

python main.py real -m real -r last_iter -k 0 1 2 -a gen -d chpoem5-train
python main.py real -m real -r last_iter -k 0 1 2 -a gen -d coco60-train
python main.py real -m real -r last_iter -k 0 1 2 -a gen -d imdb30-train
python main.py real -m real -r last_iter -k 0 1 2 -a gen -d emnlp60-train
python main.py real -m real -r last_iter -k 0 1 2 -a gen -d threecorpus75-train
python main.py real -m real -r last_iter -k 0 1 2 -a gen -d wiki72-train
python main.py oracle -m real -r last_iter -k 0 1 2 -a gen -d oracle75-train

#./code_collector.sh ipm_up
set +x
