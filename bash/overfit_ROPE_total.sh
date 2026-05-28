#!/usr/bin/env bash

#data=("10" "20" "30" "40" "50" "60" "64")
data=("$1")

#exps=(10 20 30 40 50 60 64)
exps=($1)

for i in "${!exps[@]}"
do
    # Experiment with absolute positional encoding

    # 20 epochs
    python main.py train-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/TV_ROPE_seed_42/exp_${exps[$i]}" --pos_enc rope --emb_dim 32 --seed 42 --epochs 20
    # Overfitting test
    python main.py test-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/TV_ROPE_seed_42/exp_${exps[$i]}"
    python sample/visualization.py "17_${data[$i]}" "Experimentos/TV_ROPE_seed_42/exp_${exps[$i]}"
    mv "Experimentos/TV_ROPE_seed_42/exp_${exps[$i]}/matrices" "Experimentos/TV_ROPE_seed_42/exp_${exps[$i]}/overfit_matrices_epochs20"
    mv "Experimentos/TV_ROPE_seed_42/exp_${exps[$i]}/test_tanda_17_${data[$i]}.pdf" "Experimentos/TV_ROPE_seed_42/exp_${exps[$i]}/test_17_${data[$i]}_epochs20.pdf"
    # Testing variable stems
    if  (($1 == 64)); then
        python main.py test-attention "sample/train_tanda_variable_stems.csv" "Experimentos/TV_ROPE_seed_42/exp_${exps[$i]}"
        python sample/visualization.py variable_stems "Experimentos/TV_ROPE_seed_42/exp_${exps[$i]}"
        mv "Experimentos/TV_ROPE_seed_42/exp_${exps[$i]}/matrices" "Experimentos/TV_ROPE_seed_42/exp_${exps[$i]}/test_matrices_epochs20"
        mv "Experimentos/TV_ROPE_seed_42/exp_${exps[$i]}/test_tanda_variable_stems.pdf" "Experimentos/TV_ROPE_seed_42/exp_${exps[$i]}/test_variable_stems_epochs20.pdf"
    fi
    
    # 100 epochs
    python main.py train-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/TV_ROPE_seed_42/exp_${exps[$i]}" --continue --epochs 100
    # Overfitting test
    python main.py test-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/TV_ROPE_seed_42/exp_${exps[$i]}"
    python sample/visualization.py "17_${data[$i]}" "Experimentos/TV_ROPE_seed_42/exp_${exps[$i]}"
    mv "Experimentos/TV_ROPE_seed_42/exp_${exps[$i]}/matrices" "Experimentos/TV_ROPE_seed_42/exp_${exps[$i]}/overfit_matrices_epochs100"
    mv "Experimentos/TV_ROPE_seed_42/exp_${exps[$i]}/test_tanda_17_${data[$i]}.pdf" "Experimentos/TV_ROPE_seed_42/exp_${exps[$i]}/test_17_${data[$i]}_epochs100.pdf"
    # Testing variable stems
    if (($1 == 64)); then
        python main.py test-attention "sample/train_tanda_variable_stems.csv" "Experimentos/TV_ROPE_seed_42/exp_${exps[$i]}"
        python sample/visualization.py variable_stems "Experimentos/TV_ROPE_seed_42/exp_${exps[$i]}"
        mv "Experimentos/TV_ROPE_seed_42/exp_${exps[$i]}/matrices" "Experimentos/TV_ROPE_seed_42/exp_${exps[$i]}/test_matrices_epochs100"
        mv "Experimentos/TV_ROPE_seed_42/exp_${exps[$i]}/test_tanda_variable_stems.pdf" "Experimentos/TV_ROPE_seed_42/exp_${exps[$i]}/test_variable_stems_epochs100.pdf"
    fi
done

