#!/usr/bin/env bash

# Experimentos de Variable Learning Rate con APE y ROPE con seed determinado por argumento ($1)
data=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10")

# loop for initial lr = 1e-3
exps=(1 2 3 4 5 6 7 8 9 10)

for i in "${!exps[@]}"
do
    python main.py train-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/VLR_APE_seed_$1/exp_${exps[$i]}" --pos_enc absolute --k_bias --lr 1e-3 --seed $1 --epochs 100
    python main.py test-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/VLR_APE_seed_$1/exp_${exps[$i]}"
    python sample/visualization.py "17_${data[$i]}" "Experimentos/VLR_APE_seed_$1/exp_${exps[$i]}"
    mv "Experimentos/VLR_APE_seed_$1/exp_${exps[$i]}/matrices" "Experimentos/VLR_APE_seed_$1/exp_${exps[$i]}/matrices_epochs100"
    mv "Experimentos/VLR_APE_seed_$1/exp_${exps[$i]}/test_tanda_17_${data[$i]}.pdf" "Experimentos/VLR_APE_seed_$1/exp_${exps[$i]}/test_17_${data[$i]}_epochs100.pdf"
done

for i in "${!exps[@]}"
do
    python main.py train-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/VLR_ROPE_seed_$1/exp_${exps[$i]}" --pos_enc rope --lr 1e-3 --seed $1 --epochs 100
    python main.py test-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/VLR_ROPE_seed_$1/exp_${exps[$i]}"
    python sample/visualization.py "17_${data[$i]}" "Experimentos/VLR_ROPE_seed_$1/exp_${exps[$i]}"
    mv "Experimentos/VLR_ROPE_seed_$1/exp_${exps[$i]}/matrices" "Experimentos/VLR_ROPE_seed_$1/exp_${exps[$i]}/matrices_epochs100"
    mv "Experimentos/VLR_ROPE_seed_$1/exp_${exps[$i]}/test_tanda_17_${data[$i]}.pdf" "Experimentos/VLR_ROPE_seed_$1/exp_${exps[$i]}/test_17_${data[$i]}_epochs100.pdf"
done


# loop for initial lr = 1e-4
exps=(11 12 13 14 15 16 17 18 19 20)

for i in "${!exps[@]}"
do
    python main.py train-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/VLR_APE_seed_$1/exp_${exps[$i]}" --pos_enc absolute --k_bias --lr 1e-4 --seed $1 --epochs 100
    python main.py test-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/VLR_APE_seed_$1/exp_${exps[$i]}"
    python sample/visualization.py "17_${data[$i]}" "Experimentos/VLR_APE_seed_$1/exp_${exps[$i]}"
    mv "Experimentos/VLR_APE_seed_$1/exp_${exps[$i]}/matrices" "Experimentos/VLR_APE_seed_$1/exp_${exps[$i]}/matrices_epochs100"
    mv "Experimentos/VLR_APE_seed_$1/exp_${exps[$i]}/test_tanda_17_${data[$i]}.pdf" "Experimentos/VLR_APE_seed_$1/exp_${exps[$i]}/test_17_${data[$i]}_epochs100.pdf"
done

for i in "${!exps[@]}"
do
    python main.py train-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/VLR_ROPE_seed_$1/exp_${exps[$i]}" --pos_enc rope --lr 1e-4 --seed $1 --epochs 100
    python main.py test-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/VLR_ROPE_seed_$1/exp_${exps[$i]}"
    python sample/visualization.py "17_${data[$i]}" "Experimentos/VLR_ROPE_seed_$1/exp_${exps[$i]}"
    mv "Experimentos/VLR_ROPE_seed_$1/exp_${exps[$i]}/matrices" "Experimentos/VLR_ROPE_seed_$1/exp_${exps[$i]}/matrices_epochs100"
    mv "Experimentos/VLR_ROPE_seed_$1/exp_${exps[$i]}/test_tanda_17_${data[$i]}.pdf" "Experimentos/VLR_ROPE_seed_$1/exp_${exps[$i]}/test_17_${data[$i]}_epochs100.pdf"
done

# loop for initial lr = 1e-5
exps=(21 22 23 24 25 26 27 28 29 30)

for i in "${!exps[@]}"
do
    python main.py train-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/VLR_APE_seed_$1/exp_${exps[$i]}" --pos_enc absolute --k_bias --lr 1e-5 --seed $1 --epochs 100
    python main.py test-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/VLR_APE_seed_$1/exp_${exps[$i]}"
    python sample/visualization.py "17_${data[$i]}" "Experimentos/VLR_APE_seed_$1/exp_${exps[$i]}"
    mv "Experimentos/VLR_APE_seed_$1/exp_${exps[$i]}/matrices" "Experimentos/VLR_APE_seed_$1/exp_${exps[$i]}/matrices_epochs100"
    mv "Experimentos/VLR_APE_seed_$1/exp_${exps[$i]}/test_tanda_17_${data[$i]}.pdf" "Experimentos/VLR_APE_seed_$1/exp_${exps[$i]}/test_17_${data[$i]}_epochs100.pdf"
done

for i in "${!exps[@]}"
do
    python main.py train-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/VLR_ROPE_seed_$1/exp_${exps[$i]}" --pos_enc rope --lr 1e-5 --seed $1 --epochs 100
    python main.py test-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/VLR_ROPE_seed_$1/exp_${exps[$i]}"
    python sample/visualization.py "17_${data[$i]}" "Experimentos/VLR_ROPE_seed_$1/exp_${exps[$i]}"
    mv "Experimentos/VLR_ROPE_seed_$1/exp_${exps[$i]}/matrices" "Experimentos/VLR_ROPE_seed_$1/exp_${exps[$i]}/matrices_epochs100"
    mv "Experimentos/VLR_ROPE_seed_$1/exp_${exps[$i]}/test_tanda_17_${data[$i]}.pdf" "Experimentos/VLR_ROPE_seed_$1/exp_${exps[$i]}/test_17_${data[$i]}_epochs100.pdf"
done

