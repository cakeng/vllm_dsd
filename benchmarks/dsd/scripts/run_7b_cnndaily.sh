pip install fastchat
pip install scikit-learn
mkdir bench_results
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="."

TARGET="lmsys/vicuna-7b-v1.5"
DRAFT="[ngram]"
LOOKUPMIN=2
LOOKUPMAX=8

# Baseline without SD
python benchmarks/dsd/scripts/sweep_server.py     \
                    --model $TARGET \
                    --port 10001 \
                    --dataset cnn_dailymail \
                    --result-file "7b_cnndailymail_baseline_no_sd" \
                    --request_rate_params "(1,2,4,8,16)" 

# DSD
python benchmarks/dsd/scripts/sweep_server.py     \
                    --model $TARGET   \
                    --port 10001 \
                    --speculative-model $DRAFT  \
                    --ngram-prompt-lookup-min $LOOKUPMIN \
                    --ngram-prompt-lookup-max $LOOKUPMAX \
                    --num-speculative-tokens 10 \
                    --dsd \
                    --result-file "7b_cnndailymail_dsd" \
                    --request_rate_params "(1,2,4,8,16)" \
                    --dataset cnn_dailymail

# Baseline with fixed SD
for i in 1 3 5 7
do
    python benchmarks/dsd/scripts/sweep_server.py     \
                    --model $TARGET   \
                    --port 10001 \
                    --dataset cnn_dailymail \
                    --speculative-model $DRAFT   \
                    --ngram-prompt-lookup-min $LOOKUPMIN \
                    --ngram-prompt-lookup-max $LOOKUPMAX \
                    --num-speculative-tokens $i \
                    --result-file "7b_cnndailymail_baseline_sd" \
                    --request_rate_params "(1,2,4,8,16)"
done