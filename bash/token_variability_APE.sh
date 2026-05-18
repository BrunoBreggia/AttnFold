#!/usr/bin/env bash

# Experimentos de Token Variability con Absolute Positional Encoding (TV_APE) con seed determinado por argumento ($1)
data=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10")

# Experiments with absolute positional embedding
exps=(1 2 3 4 5 6 7 8)

# loop for 32 dim embedding
for i in "${!exps[@]}"
do
    python main.py train-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}" --pos_enc absolute --mask --k_bias --emb_dim 32 --seed $1 --epochs 20
    python main.py test-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}"
    python sample/visualization.py "17_${data[$i]}" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}"
    mv "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/matrices" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/matrices_epochs20"
    mv "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/test_tanda_17_${data[$i]}.pdf" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/test_17_${data[$i]}_epochs20.pdf"

    python main.py train-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}" --continue --epochs 100
    python main.py test-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}"
    python sample/visualization.py "17_${data[$i]}" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}"
    mv "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/matrices" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/matrices_epochs100"
    mv "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/test_tanda_17_${data[$i]}.pdf" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/test_17_${data[$i]}_epochs100.pdf"
done


# loop for 64 dim embedding
for i in "${!exps[@]}"
do
    python main.py train-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}" --pos_enc absolute --mask --k_bias --emb_dim 64 --seed $1 --epochs 20
    python main.py test-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}"
    python sample/visualization.py "17_${data[$i]}" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}"
    mv "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/matrices" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/matrices_epochs20"
    mv "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/test_tanda_17_${data[$i]}.pdf" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/test_17_${data[$i]}_epochs20.pdf"

    python main.py train-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}" --continue --epochs 100
    python main.py test-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}"
    python sample/visualization.py "17_${data[$i]}" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}"
    mv "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/matrices" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/matrices_epochs100"
    mv "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/test_tanda_17_${data[$i]}.pdf" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/test_17_${data[$i]}_epochs100.pdf"
done


# loop for 128 dim embedding
for i in "${!exps[@]}"
do
    python main.py train-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}" --pos_enc absolute --mask --k_bias --emb_dim 128 --seed $1 --epochs 20
    python main.py test-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}"
    python sample/visualization.py "17_${data[$i]}" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}"
    mv "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/matrices" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/matrices_epochs20"
    mv "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/test_tanda_17_${data[$i]}.pdf" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/test_17_${data[$i]}_epochs20.pdf"

    python main.py train-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}" --continue --epochs 100
    python main.py test-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}"
    python sample/visualization.py "17_${data[$i]}" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}"
    mv "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/matrices" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/matrices_epochs100"
    mv "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/test_tanda_17_${data[$i]}.pdf" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/test_17_${data[$i]}_epochs100.pdf"
done


# loop for 256 dim embedding
for i in "${!exps[@]}"
do
    python main.py train-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}" --pos_enc absolute --mask --k_bias --emb_dim 256 --seed $1 --epochs 20
    python main.py test-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}"
    python sample/visualization.py "17_${data[$i]}" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}"
    mv "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/matrices" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/matrices_epochs20"
    mv "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/test_tanda_17_${data[$i]}.pdf" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/test_17_${data[$i]}_epochs20.pdf"

    python main.py train-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}" --continue --epochs 100
    python main.py test-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}"
    python sample/visualization.py "17_${data[$i]}" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}"
    mv "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/matrices" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/matrices_epochs100"
    mv "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/test_tanda_17_${data[$i]}.pdf" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/test_17_${data[$i]}_epochs100.pdf"
done


# loop for 512 dim embedding
for i in "${!exps[@]}"
do
    python main.py train-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}" --pos_enc absolute --mask --k_bias --emb_dim 512 --seed $1 --epochs 20
    python main.py test-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}"
    python sample/visualization.py "17_${data[$i]}" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}"
    mv "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/matrices" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/matrices_epochs20"
    mv "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/test_tanda_17_${data[$i]}.pdf" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/test_17_${data[$i]}_epochs20.pdf"

    python main.py train-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}" --continue --epochs 100
    python main.py test-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}"
    python sample/visualization.py "17_${data[$i]}" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}"
    mv "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/matrices" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/matrices_epochs100"
    mv "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/test_tanda_17_${data[$i]}.pdf" "Experimentos/TV_APE_seed_$1/exp_${exps[$i]}/test_17_${data[$i]}_epochs100.pdf"
done
