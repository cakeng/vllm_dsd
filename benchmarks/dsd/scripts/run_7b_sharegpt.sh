export CUDA_VISIBLE_DEVICES=0

TARGET="lmsys/vicuna-7b-v1.5"
DRAFT="eqhylxx/vicuna-160m"

# Baseline without SD
python benchmarks/dsd/scripts/sweep_server.py     \
                    --model $TARGET \
                    --port 10001

# Baseline with fixed SD
for i in 1 3 5 7
do
    python benchmarks/dsd/scripts/sweep_server.py     \
                    --model $TARGET   \
                    --speculative-model $DRAFT   \
                    --num-speculative-tokens $i \
                    --port 10001
done

# DSD
python benchmarks/dsd/scripts/sweep_server.py     \
                    --model $TARGET   \
                    --speculative-model $DRAFT  \
                    --num-speculative-tokens 8 \
                    --dsd \
                    --port 10001

