#!/usr/bin/env bash
user="danial"
mll_V_coco="$user@192.168.197.189:codes/evaluator/"
mll_V_emnlp="$user@192.168.197.189:codes/evaluator_emnlp/"
mll_phd="/home/danial/newhome/codes/evaluator/"
mll_phd_three="/home/danial/newhome/codes/evaluator_threecorpus/"
mll_master="$user@192.168.207.168:Codes/evaluator/"
mll_undergrad_emnlp="$user@192.168.207.167:Codes/evaluator/"
ipm="montahaie@cluster.hpc.ipm.ac.ir:codes/evaluator/data/temp_models/"
data_path="data/*"
destination="./data"

run_command () {
    if [[ $1 == *@* ]]; then
        echo "going to scp"
        scp -r "$1$data_path" "$destination"
    else
        echo "going to cp"
        cp -r $1$data_path $destination
    fi
}

if [ ! -d $destination ]; then
    mkdir "$destination"
    echo "new dir $destination created"
fi
if [ $1 = "v_coco" ]; then
    run_command "$mll_V_coco"
elif [ $1 = "v_emnlp" ]; then
    run_command "$mll_V_emnlp"
elif [ $1 = "phd" ]; then
    run_command "$mll_phd"
elif [ $1 = "phd_three" ]; then
    run_command "$mll_phd_three"
elif [ $1 = "under_emnlp" ]; then
    run_command "$mll_undergrad_emnlp"
elif [ $1 = "ipm" ]; then
#    find data/temp_models/ -name *.zip -exec rm {} ';'
#    find data/temp_models/ -name best_history.json -exec rm {} ';'
    rsync --include "*.json" -vau data/temp_models/ $ipm
#    scp -r data/temp_models/ $ipm
else
    echo "invalid input!!"
fi