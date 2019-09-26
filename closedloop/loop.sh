#!/bin/bash

## A closed loop shell which does
## 1) call mohex-self-play produce a set of games
## 2) refine neural net on those games by running ../simplefeature3/neuralnet/resnet.py  
## 3) freeze trained neural net via ../simplefeature3/freeze_graph/freeze_graph_main.py
## 4) use nn_load to load new nn model 
## 5) Go back to step 1

## Since step 1) is rather time and computation consuming,
## e.g., 20-block version AlphaGo Zero generated over 4 million games.
## How fast is mohex-self-play on a single GPU GTX1080 computer? 
## If 1000 simulations per move, a move consumes about 1s 
## So if a game has 60 moves, then it takes 1*10^6 minutes for 1 million game
## 1 million minutes is 10**6/60.0 ~= 700 days

# We do not start from random weights and 'near' random games (note that 
# the games generated by MCTS+random weighted NN are not entirely random!
# Because of the tree search, at least moves near the end are not random,
# --- that's why bootstrapping worked! Learning from pure random games gain nothing but random!) 
# We start from a pre-trained net on MoHex 2.0 self-play games. 
# We first generated a set of games; learning begins from those games.

max_number() {
    printf "%s\n" "$@" | sort -rg | head -n1
}


if [ $# -lt 7 ] ; then
    echo "Usage: $0 mohex_exe numberOfGamesPerWokerPerIteration numberOfIterations option[selfplay|selfplaytrain] numOfParallelProcess boardsize search_parameter_config_file[*.htp]"
    echo "Note: games will be saved at storage/, named as boardsizexboardsize_simPerMove_games.txt, nn model will be saved at selfplayNN/ "
    exit 1
fi 

mohex_exe=$1 
n_games_per_ite=$2 #eg 1000 games per iteration per worker
total_iteration=$3 #eg 10
runoption=$4 
#e.g., 10 instances in parallel
Npp=$5 
boardsize=$6
search_config=$7

n_sim_per_move=$(grep 'param_mohex max_games' $search_config | grep -oE '[0-9]+')
echo "n_sim_per_move: $n_sim_per_move"
dithered_threshold=$(grep 'param_mohex moveselect_ditherthreshold' $search_config | grep -oE '[0-9]+')
echo "dithered_threshold: $dithered_threshold"

game_file_name=${boardsize}x${boardsize}_mcts${n_sim_per_move}_games.txt

if [[ $boardsize -gt 19 || $boardsize -lt 5 ]]; then
    echo "boardsize should between [5,19], input is too large or too small"
    exit 1
fi


if [ $runoption != "selfplay" ] && [ $runoption != "selfplaytrain" ] ; then
    echo "Usage: $0 mohex_exe numberOfSimulationsPerMove numberOfGamesPerIteration numberOfIterations option[selfplay|selfplaytrain] numOfParallelProcess boardsize"
    echo "option could only be selfplay or selfplaytrain"
    echo "selfplay: only generate selfplay games"
    echo "selfplaytrain: generate games and then train the net again"
    exit 1
fi

save_dir=./storage
mkdir -p $save_dir
nn_model_dir=./selfplayNN
mkdir -p $nn_model_dir
constant_nn_model_dir=./constant_nn/
mkdir -p $constant_nn_model_dir

log_dir=./log_games
mkdir -p $log_dir

INF=10000000

n_train_games=$((n_games_per_ite*Npp)) #num of train games is number of wokers * num games per worker per iteration
#below are nn parameters
epoch_limit_per_train=5
n_blocks=10
n_filters_per_layer=32
fc_q_head=1 #1 with fully-connected q head, 0 otherwise
fc_p_head=0 #1 with fully-connected p head, 0 otherwise
l2_regularize=0.0001
lr_init=0.005
optimizer="momentum"
lr_decay=0.90
lr_min=0.000005

v_coeff=1.0
q_coeff=1.0 #change 0.0 for 2HNN
q_all_coeff=1.0 # 0.0 for 2HNN
value_penalty_weight=1.0 # 0.0 for 2HNN

#post_process="only_shuffle" # used for 2HNN
post_process="compute_average" 


rm config*.htp 2>/dev/null && echo "remove obsolete config files"

n_gpus=0
type nvidia-smi >/dev/null 2>&1
if [[ $? -eq 0 ]]; then
    n_gpus=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l 2>/dev/null`
fi

if [[ -z $n_gpus ]]; then
    n_gpus=0
fi
echo "num of gpus: $n_gpus"

echo "====search parameters===="
cat $search_config
echo "========================="

echo "========
boardsize: $boardsize x $boardsize
Loop: $total_iteration total iterations
mode: $runoption 
$Npp selfplay instances in parallel 
$n_games_per_ite games per worker each iteration  

selfplay PV-MCTS:
$n_sim_per_move simulations per move 

NN configuration: 
n_blocks: $n_blocks
n_filters_per_layer: $n_filters_per_layer
fc_q_head?: $fc_q_head
fc_p_head?: $fc_p_head

Train NN:
each train epoch_limit: $epoch_limit_per_train;
each train games is the product of n wokers and n games per worker: $n_train_games
optimizer: $optimizer
initial learning rate: $lr_init
learning rate iteration decay ratio: $lr_decay
l2_regularize: $l2_regularize
========
"

export PYTHONPATH=../simplefeature3/
if [ -z $game_file_name ] ; then
    game_file_name=${boardsize}x${boardsize}_selfplay_games.txt
fi

selfplay() {
    echo -e "mohex-self-play: $1 processes in parallel"
    hname=`hostname`
    N=$1
    for((i=1; i<=$N; i++))
    do
        if [[ $n_gpus -gt 0 ]]; then 
            ii=$((i-1)) 
            export CUDA_VISIBLE_DEVICES=$((ii % n_gpus))
            echo $CUDA_VISIBLE_DEVICES
        fi
        SEED=$(($RANDOM))
        $mohex_exe --seed $SEED <$2 >$log_dir/${hname}_selfplay${i}_log.out 2>&1 &
    done
    unset CUDA_VISIBLE_DEVICES
    wait 
}

for((ite=0; ite<=$total_iteration; ite++))
do
    echo "====iteration:$ite===="
    echo "load most recent nn model at $nn_model_dir"
    previous_pb=""
    nn_basename=""
    nn_to_load=""
    ls $nn_model_dir/*.pb 1>/dev/null 2>/dev/null
    if [[ $? -eq 0 ]] ; then
        previous_pb=`ls -t $nn_model_dir/*.pb | head -n1 2>/dev/null`
        cp $previous_pb $constant_nn_model_dir
        nn_basename=`basename $previous_pb`
        nn_to_load=`(cd $(dirname $previous_pb) && (pwd))`/`basename $previous_pb`
    fi
    echo "loaded $nn_basename"
    echo "Iteration $ite of $total_iteration, using MCTS-$nn_basename selfplay game generation"

    gamefile=$game_file_name
    gamefilepath=${save_dir}/$gamefile

    common_config=`cat $search_config`
    echo "$common_config" >>config$ite.htp
    echo -e "nn_load $nn_to_load
    nn_load
    boardsize $boardsize 
    param_mohex
    mohex-self-play $n_games_per_ite $gamefilepath 
    quit" >> config$ite.htp
    selfplay $Npp  config$ite.htp
    wait 

    if [ $runoption == "selfplay" ] ; then
        continue
    fi

    echo "========"
    train_file=${save_dir}/${boardsize}x${boardsize}train$ite.txt
    echo "select most recent $n_train_games selfplay games from $gamefilepath, dump to $train_file"
    tail -$n_train_games  $gamefilepath > $train_file
    echo "extracting games to examples"
    python extractor.py --post_process=$post_process --input_file=$train_file --output_file=$train_file.out --boardsize=$boardsize --dithered_threshold=$dithered_threshold
    wait

    previous_checkpoint=""
    ls -t $nn_model_dir/${boardsize}x${boardsize}*.meta 1>/dev/null 2>/dev/null
    if [[ $? -eq 0 ]]; then
        echo "remove .meta"
        previous_checkpoint=`ls -t $nn_model_dir/${boardsize}x${boardsize}*.meta | head -n1 2>/dev/null`
        suffix=".meta"
        previous_checkpoint=${previous_checkpoint%$suffix}; #Remove suffix
    fi

    echo "========"
    echo -e "Training the neural net on $train_file.out-post
    save nn models to $nn_model_dir, 
    logs in /tmp/train_logs.out, 
    training model initialized from old model $previous_checkpoint"
    lr=$lr_init
    y=$((ite + 1)) #decay**(ite+1)
    lr=`echo "$lr*e($y*l($lr_decay))" | bc -l` #x^y = e^{y*log(x)}
    lr=$(max_number $lr $lr_min)
    echo "learning_rate: $lr"
    with_fc_q_head=""
    with_fc_p_head=""
    if [[ $fc_q_head -gt 0 ]] ; then
        echo "fc q head"
        with_fc_q_head="--fc_q_head"
    fi
    if [[ $fc_p_head -gt 0 ]] ; then
        echo "fc p head"
        with_fc_p_head="--fc_p_head"
    fi
    input_file_dithered=${train_file}.out_dithered-post
    input_file_normal=${train_file}.out-post
    if [[ -f $input_file_dithered && -s $input_file_dithered ]]; then
        echo "dithered not empty, train on it first, action value augmentation turned off (--q_all_coeff=0.0)"
        python ../simplefeature3/neuralnet/resnet.py --verbose --n_hidden_blocks=$n_blocks --label_type='prob' --input_file=$input_file_dithered --output_dir=$nn_model_dir/ --max_train_step=$INF --resume_train --previous_checkpoint=$previous_checkpoint --epoch_limit=$epoch_limit_per_train --boardsize=$boardsize  --l2_weight=$l2_regularize --n_filters_per_layer=$n_filters_per_layer --optimizer=$optimizer --learning_rate=$lr $with_fc_q_head $with_fc_p_head --v_coeff=$v_coeff --q_coeff=$q_coeff --q_all_coeff=0.0 --value_penalty_weight=$value_penalty_weight > /tmp/train_logs_dithered.out 2>/tmp/train_log_dithered.err 

        previous_checkpoint=`ls -t $nn_model_dir/${boardsize}x${boardsize}*.meta | head -n1 2>/dev/null`
        suffix=".meta"
        previous_checkpoint=${previous_checkpoint%$suffix}; #Remove suffix
    fi
    python ../simplefeature3/neuralnet/resnet.py --verbose --n_hidden_blocks=$n_blocks --label_type='prob' --input_file=$input_file_normal --output_dir=$nn_model_dir/ --max_train_step=$INF --resume_train --previous_checkpoint=$previous_checkpoint --epoch_limit=$epoch_limit_per_train --boardsize=$boardsize  --l2_weight=$l2_regularize --n_filters_per_layer=$n_filters_per_layer --optimizer=$optimizer --learning_rate=$lr $with_fc_q_head $with_fc_p_head --v_coeff=$v_coeff --q_coeff=$q_coeff --q_all_coeff=$q_all_coeff --value_penalty_weight=$value_penalty_weight > /tmp/train_logs.out 2>/tmp/train_logs.err 

    echo "========="
    ckpt=$nn_model_dir/${boardsize}x${boardsize}train$ite.txt.out-post.ckpt-$epoch_limit_per_train
    graph=$nn_model_dir/resnet_evaluate_${n_blocks}_${n_filters_per_layer}.pbtext
    if [ -f "${graph}" ] ; then
        echo "graph definition $graph found"
    else 
        echo "no existing graph $graph, creating by running evaluation on the training dataset" 
        python ../simplefeature3/neuralnet/resnet.py --verbose --n_hidden_blocks=$n_blocks --input_file=${train_file}.out-post --evaluate --previous_checkpoint=$previous_checkpoint $with_fc_q_head $with_fc_p_head --n_filters_per_layer=$n_filters_per_layer >/tmp/evaluate.out 2>/tmp/evaluate.err 
        mv resnet_evaluate_${n_blocks}_${n_filters_per_layer}.pbtext $graph
    fi
    echo "converting neural net model to constant graph and save to $nn_model_dir"
    python ../simplefeature3/freeze_graphs/freeze_graph_main.py --input_graph=$graph --checkpoint=$ckpt >/tmp/freeze_log.txt  2>&1
    echo "ite $ite finished"
    echo "================"
done
