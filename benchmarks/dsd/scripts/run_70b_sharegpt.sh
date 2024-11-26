export CUDA_VISIBLE_DEVICES=0,1,2,3

TARGET="meta-llama/Llama-3.1-70B-Instruct"
DRAFT="meta-llama/Llama-3.2-1B-Instruct"
port=$1

# Baseline without SD
python benchmarks/dsd/scripts/sweep_server.py     \
                    --model $TARGET \
                    --port $port \
                    --result-file "70b_sharegpt_baseline_no_sd" \
                    --request_rate_params "(1,10,2)"

# DSD
python benchmarks/dsd/scripts/sweep_server.py     \
                    --model $TARGET  \
                    --speculative-model $DRAFT   \
                    --num-speculative-tokens 7 \
                    --dsd \
                    --port $(($port + 1)) \
                    --result-file "70b_sharegpt_dsd" \
                    --request_rate_params "(1,10,2)"

# Baseline with fixed SD
for i in 1 3 5 7
do
    python benchmarks/dsd/scripts/sweep_server.py     \
                    --model $TARGET   \
                    --speculative-model $DRAFT   \
                    --num-speculative-tokens $i \
                    --port $(($port + i + 1)) \
                    --result-file "70b_sharegpt_baseline_sd" \
                    --request_rate_params "(1,10,2)"
done


