pip install fastchat
pip install scikit-learn
mkdir bench_results
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="."

TARGET="lmsys/vicuna-7b-v1.5"
DRAFT="eqhylxx/vicuna-160m"
port=$1

# Baseline without SD
python benchmarks/dsd/scripts/sweep_server.py     \
                    --dataset sonnet \
                    --model $TARGET \
                    --port $port \
                    --result-file "7b_sonnet_baseline_no_sd" \
                    --request_rate_params "(1,2,4,8,16)"

# DSD
python benchmarks/dsd/scripts/sweep_server.py     \
                    --dataset sonnet \
                    --model $TARGET   \
                    --speculative-model $DRAFT  \
                    --num-speculative-tokens 5 \
                    --dsd \
                    --port $(($port + 1)) \
                    --result-file "7b_sonnet_dsd" \
                    --request_rate_params "(1,2,4,8,16)"

# Baseline with fixed SD
for i in 1 3 5 7
do
    python benchmarks/dsd/scripts/sweep_server.py     \
                    --dataset sonnet \
                    --model $TARGET   \
                    --speculative-model $DRAFT   \
                    --num-speculative-tokens $i \
                    --port $(($port + i + 1)) \
                    --result-file "7b_sonnet_baseline_sd" \
                     --request_rate_params "(1,2,4,8,16)"
done


