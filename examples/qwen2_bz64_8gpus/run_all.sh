range="0 1 2 3 4"

for seed in ${range}; do

    for method in pm_et-m52_ucb bo-m52_ucb xgb cost random; do
        python run.py $method $seed
    done

done


for seed in ${range}; do

    for bovar in pm_et pm bo_et bo; do
        for kern in dkm52 m52; do
            for acq in ucb; do
                python run.py ${bovar}-${kern}_${acq} $seed
            done
        done
    done

done
