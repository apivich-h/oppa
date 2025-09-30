range="1"

for seed in ${range}; do

    for method in bo-m52_ucb xgb; do
        python -W ignore run.py $method $seed
    done

done

