export CUDA_VISIBLE_DEVICES=4,5,6,7

TARGET="meta-llama/Llama-3.1-70B-Instruct"
DRAFT="meta-llama/Llama-3.2-1B-Instruct"

# Baseline without SD
python benchmarks/dsd/scripts/sweep_server.py     \
                    --model $TARGET \
                    --port 10000 \
                    --result-file "70b_sharegpt_baseline_no_sd"

# Baseline with fixed SD
for i in 1 3 5 7
do
    python benchmarks/dsd/scripts/sweep_server.py     \
                    --model $TARGET   \
                    --speculative-model $DRAFT   \
                    --num-speculative-tokens $i \
                    --port 10000 \
                    --result-file "70b_sharegpt_baseline_sd"
done

# DSD
python benchmarks/dsd/scripts/sweep_server.py     \
                    --model $TARGET  \
                    --speculative-model $DRAFT   \
                    --num-speculative-tokens 8 \
                    --dsd \
                    --port 10000 \
                    --result-file "70b_sharegpt_dsd"

