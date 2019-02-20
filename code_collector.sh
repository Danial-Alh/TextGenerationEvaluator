#!/usr/bin/env bash
user="danial"
mll_V="$user@192.168.197.189:codes/evaluator/"
mll_phd="/home/danial/newhome/codes/evaluator/"
mll_phd_three="/home/danial/newhome/codes/evaluator_threecorpus/"
mll_master="$user@192.168.207.168:Codes/evaluator/"
mll_undergrad="$user@192.168.207.167:codes/evaluator/"
data_path="data/*"
destination="./data"

run_command () {
    if [[ $1 == danial* ]]; then
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
if [ $1 = "v" ]; then
    run_command "$mll_V"
elif [ $1 = "phd" ]; then
    run_command "$mll_phd"
elif [ $1 = "phd_three" ]; then
    run_command "$mll_phd_three"
elif [ $1 = "undergrad" ]; then
    run_command "$mll_undergrad"
fi
