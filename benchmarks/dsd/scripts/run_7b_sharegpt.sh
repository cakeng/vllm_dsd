export CUDA_VISIBLE_DEVICES=0

# Baseline without SD
python benchmarks/dsd/scripts/sweep_server.py     \
                    --model lmsys/vicuna-7b-v1.5 \
                    --port 10001

# Baseline with fixed SD
for i in 1 3 5 7
do
    python benchmarks/dsd/scripts/sweep_server.py     \
                    --model lmsys/vicuna-7b-v1.5   \
                    --speculative-model eqhylxx/vicuna-160m   \
                    --num-speculative-tokens $i \
                    --port 10001
done

# DSD
python benchmarks/dsd/scripts/sweep_server.py     \
                    --model lmsys/vicuna-7b-v1.5   \
                    --speculative-model eqhylxx/vicuna-160m   \
                    --num-speculative-tokens 8 \
                    --dsd \
                    --port 10001

