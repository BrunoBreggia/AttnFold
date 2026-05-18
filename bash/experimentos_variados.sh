#!/usr/bin/env bash

data=("1" "2" "3" "4")

# Experiments with absolute positional embedding
exps=(125 126 127 128)
for i in "${!exps[@]}"
do
    python main.py train-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/exp_${exps[$i]}" --pos_enc absolute --mask --k_bias --epochs 20
    python main.py test-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/exp_${exps[$i]}"
    python sample/visualization.py "17_${data[$i]}" "Experimentos/exp_${exps[$i]}"
    mv "Experimentos/exp_${exps[$i]}/matrices" "Experimentos/exp_${exps[$i]}/matrices_epochs20"
    mv "Experimentos/exp_${exps[$i]}/test_tanda_17_${data[$i]}.pdf" "Experimentos/exp_${exps[$i]}/test_17_${data[$i]}_epochs20.pdf"

    python main.py train-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/exp_${exps[$i]}" --continue --epochs 100
    python main.py test-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/exp_${exps[$i]}"
    python sample/visualization.py "17_${data[$i]}" "Experimentos/exp_${exps[$i]}"
    mv "Experimentos/exp_${exps[$i]}/matrices" "Experimentos/exp_${exps[$i]}/matrices_epochs100"
    mv "Experimentos/exp_${exps[$i]}/test_tanda_17_${data[$i]}.pdf" "Experimentos/exp_${exps[$i]}/test_17_${data[$i]}_epochs100.pdf"
done

for i in "${!exps[@]}"
do
    python main.py train-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/exp_${exps[$i]}" --continue --epochs 500
    python main.py test-attention "sample/train_tanda_17_${data[$i]}.csv" "Experimentos/exp_${exps[$i]}"
    python sample/visualization.py "17_${data[$i]}" "Experimentos/exp_${exps[$i]}"
    mv "Experimentos/exp_${exps[$i]}/matrices" "Experimentos/exp_${exps[$i]}/matrices_epochs500"
    mv "Experimentos/exp_${exps[$i]}/test_tanda_17_${data[$i]}.pdf" "Experimentos/exp_${exps[$i]}/test_17_${data[$i]}_epochs500.pdf"
done

# Experiments with rotary positional embedding
# exps=(122)
# for i in "${!exps[@]}"
# do
#     python main.py train-attention "sample/train_tanda_16_testeo.csv" "Experimentos/exp_${exps[$i]}" --pos_enc rope --epochs 20
#     python main.py test-attention sample/train_tanda_16_testeo.csv "Experimentos/exp_${exps[$i]}"
#     python sample/visualization.py 16_testeo "Experimentos/exp_${exps[$i]}"
#     mv "Experimentos/exp_${exps[$i]}/matrices" "Experimentos/exp_${exps[$i]}/matrices_epochs20"
#     mv "Experimentos/exp_${exps[$i]}/test_tanda_16_testeo.pdf" "Experimentos/exp_${exps[$i]}/test_16_testeo_epochs20.pdf"

#     python main.py train-attention "sample/train_tanda_16_testeo.csv" "Experimentos/exp_${exps[$i]}" --continue --epochs 100
#     python main.py test-attention sample/train_tanda_16_testeo.csv "Experimentos/exp_${exps[$i]}"
#     python sample/visualization.py 16_testeo "Experimentos/exp_${exps[$i]}"
#     mv "Experimentos/exp_${exps[$i]}/matrices" "Experimentos/exp_${exps[$i]}/matrices_epochs100"
#     mv "Experimentos/exp_${exps[$i]}/test_tanda_16_testeo.pdf" "Experimentos/exp_${exps[$i]}/test_16_testeo_epochs100.pdf"

#     python main.py train-attention "sample/train_tanda_16_testeo.csv" "Experimentos/exp_${exps[$i]}" --continue --epochs 500
#     python main.py test-attention sample/train_tanda_16_testeo.csv "Experimentos/exp_${exps[$i]}"
#     python sample/visualization.py 16_testeo "Experimentos/exp_${exps[$i]}"
#     mv "Experimentos/exp_${exps[$i]}/matrices" "Experimentos/exp_${exps[$i]}/matrices_epochs500"
#     mv "Experimentos/exp_${exps[$i]}/test_tanda_16_testeo.pdf" "Experimentos/exp_${exps[$i]}/test_16_testeo_epochs500.pdf"
# done
