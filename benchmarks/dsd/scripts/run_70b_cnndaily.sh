export CUDA_VISIBLE_DEVICES=4,5,6,7

TARGET="meta-llama/Llama-3.1-70B-Instruct"
DRAFT="[ngram]"
LOOKUPMIN=2
LOOKUPMAX=8
port=$1

# Baseline without SD
python benchmarks/dsd/scripts/sweep_server.py     \
                    --model $TARGET \
                    --port $port \
                    --dataset cnn_dailymail \
                    --result-file "70b_cnndailymail_baseline_no_sd" \
                    --request_rate_params "(2,8,2)"

# DSD
python benchmarks/dsd/scripts/sweep_server.py     \
                    --model $TARGET   \
                    --port $(($port + 1)) \
                    --dataset cnn_dailymail \
                    --speculative-model $DRAFT  \
                    --ngram-prompt-lookup-min $LOOKUPMIN \
                    --ngram-prompt-lookup-max $LOOKUPMAX \
                    --num-speculative-tokens 7 \
                    --dsd \
                    --result-file "70b_cnndailymail_dsd" \
                    --request_rate_params "(2,8,2)"

# Baseline with fixed SD
for i in 3 5 7
do
    python benchmarks/dsd/scripts/sweep_server.py     \
                    --model $TARGET   \
                    --port $(($port + i + 1)) \
                    --dataset cnn_dailymail \
                    --speculative-model $DRAFT   \
                    --ngram-prompt-lookup-min $LOOKUPMIN \
                    --ngram-prompt-lookup-max $LOOKUPMAX \
                    --num-speculative-tokens $i \
                    --result-file "70b_cnndailymail_baseline_sd" \
                    --request_rate_params "(2,8,2)"
done