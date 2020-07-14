#!/usr/bin/env bash
run=$1

CUDA_VISIBLE_DEVICES=0 python3 main.py real -d chpoem5-train -a eval -m real -r last_iter -run 0 1 2
CUDA_VISIBLE_DEVICES=0 python3 main.py real -d coco60-train -a eval -m real -r last_iter -run 0 1 2
CUDA_VISIBLE_DEVICES=0 python3 main.py real -d imdb30-train -a eval -m real -r last_iter -run 0 1 2
CUDA_VISIBLE_DEVICES=0 python3 main.py real -d emnlp60-train -a eval -m real -r last_iter -run 0 1 2
CUDA_VISIBLE_DEVICES=0 python3 main.py real -d threecorpus75-train -a eval -m real -r last_iter -run 0 1 2
CUDA_VISIBLE_DEVICES=0 python3 main.py real -d wiki72-train -a eval -m real -r last_iter -run 0 1 2
CUDA_VISIBLE_DEVICES=0 python3 main.py oracle -d oracle75-train -a eval -m real -r last_iter -run 0 1 2
