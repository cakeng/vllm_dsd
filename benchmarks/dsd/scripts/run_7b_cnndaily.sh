export CUDA_VISIBLE_DEVICES=0

TARGET="lmsys/vicuna-7b-v1.5"
DRAFT="[ngram]"
LOOKUPMIN=2
LOOKUPMAX=8

# # Baseline without SD
# python benchmarks/dsd/scripts/sweep_server.py     \
#                     --model $TARGET \
#                     --port 10001 \
#                     --dataset cnn_dailymail \
#                     --result-file "7b_cnndailymail_baseline_no_sd"

# # DSD
# python benchmarks/dsd/scripts/sweep_server.py     \
#                     --model $TARGET   \
#                     --port 10001 \
#                     --speculative-model $DRAFT  \
#                     --ngram-prompt-lookup-min $LOOKUPMIN \
#                     --ngram-prompt-lookup-max $LOOKUPMAX \
#                     --num-speculative-tokens 8 \
#                     --dsd \
#                     --result-file "7b_cnndailymail_dsd"

# Baseline with fixed SD
for i in 5
do
    python benchmarks/dsd/scripts/sweep_server.py     \
                    --model $TARGET   \
                    --port 10001 \
                    --dataset cnn_dailymail \
                    --speculative-model $DRAFT   \
                    --ngram-prompt-lookup-min $LOOKUPMIN \
                    --ngram-prompt-lookup-max $LOOKUPMAX \
                    --num-speculative-tokens $i \
                    --result-file "7b_cnndailymail_baseline_sd"
done