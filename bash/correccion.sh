#!/usr/bin/env bash

data=("2" "3" "4")

# Experiments with absolute positional embedding
exps=(126 127 128)
for i in "${!exps[@]}"
do
    # rm "Experimentos/exp_${exps[$i]}/test_17_1_epochs20.pdf"
    # rm "Experimentos/exp_${exps[$i]}/test_17_1_epochs100.pdf"
    # rm "Experimentos/exp_${exps[$i]}/test_17_1_epochs500.pdf"

    cp -r "Experimentos/exp_${exps[$i]}/matrices_epochs20" "Experimentos/exp_${exps[$i]}/matrices"
    python sample/visualization.py "17_${data[$i]}" "Experimentos/exp_${exps[$i]}"
    mv "Experimentos/exp_${exps[$i]}/test_tanda_17_${data[$i]}.pdf" "Experimentos/exp_${exps[$i]}/test_17_${data[$i]}_epochs20.pdf"
    rm -r "Experimentos/exp_${exps[$i]}/matrices"

    cp -r "Experimentos/exp_${exps[$i]}/matrices_epochs100" "Experimentos/exp_${exps[$i]}/matrices"
    python sample/visualization.py "17_${data[$i]}" "Experimentos/exp_${exps[$i]}"
    mv "Experimentos/exp_${exps[$i]}/test_tanda_17_${data[$i]}.pdf" "Experimentos/exp_${exps[$i]}/test_17_${data[$i]}_epochs100.pdf"
    rm -r "Experimentos/exp_${exps[$i]}/matrices"

    cp -r "Experimentos/exp_${exps[$i]}/matrices_epochs500" "Experimentos/exp_${exps[$i]}/matrices"
    python sample/visualization.py "17_${data[$i]}" "Experimentos/exp_${exps[$i]}"
    mv "Experimentos/exp_${exps[$i]}/test_tanda_17_${data[$i]}.pdf" "Experimentos/exp_${exps[$i]}/test_17_${data[$i]}_epochs500.pdf"
    rm -r "Experimentos/exp_${exps[$i]}/matrices"
done

