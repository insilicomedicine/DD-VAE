set -ex
mkdir -p bo/results/$2/
python bo/gen_latent.py \
        --data data/zinc/250k_rndm_zinc_drugs_clean.smi.gz \
        --model $1 \
        --device $3 --save_dir bo/results/$2/

for SEED in $(seq 1 10)
do
    mkdir -p bo/results/$2/experiment\_$SEED/
    python bo/run_bo.py \
           --model $1 \
           --save_dir bo/results/$2/experiment\_$SEED/ \
           --device $3 \
           --seed $SEED \
           --load_dir bo/results/$2/ > bo/results/$2/experiment\_$SEED/log.txt &
    sleep 20
done
