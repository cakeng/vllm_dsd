TARGET="lmsys/vicuna-7b-v1.5"
input_len=256

for batch_size in 1 2 4 8 16 32 64
do
    python benchmarks/benchmark_latency.py \
        --model  $TARGET \
        --input-len $input_len \
        --output-len 256 \
        --batch-size $batch_size \
        --num-iters-warmup 5 \
        --num-iters 10 \
        --output-json benchmarks/dsd/results/llama2-7b-ngram/org_bz=${batch_size}_input-len=${input_len}.json
done


for batch_size in 1 2 4 8 16 32 64
do
    python benchmarks/benchmark_latency.py \
        --model  $TARGET \
        --input-len $input_len \
        --output-len 256 \
        --batch-size $batch_size \
        --num-iters-warmup 5 \
        --num-iters 10 \
        --speculative-model "[ngram]" \
        --ngram-prompt-lookup-min 2 \
        --ngram-prompt-lookup-max 8 \
        --num-speculative-tokens 10 \
        --dsd \
        --acceptance-rate 0.7 \
        --dummy-match 0.5 \
        --output-json benchmarks/dsd/results/llama2-7b-ngram/dsd_bz=${batch_size}_input-len=${input_len}.json
done


for batch_size in 1 2 4 8 16 32 64
do
    for num_speculative_tokens in 1 3 5 7 9
    do
        python benchmarks/benchmark_latency.py \
            --model  $TARGET \
            --input-len $input_len \
            --output-len 256 \
            --batch-size $batch_size \
            --num-iters-warmup 5 \
            --num-iters 10 \
            --speculative-model "[ngram]" \
            --ngram-prompt-lookup-min 2 \
            --ngram-prompt-lookup-max 8 \
            --num-speculative-tokens $num_speculative_tokens \
            --acceptance-rate 0.7 \
            --dummy-match 0.5 \
            --output-json benchmarks/dsd/results/llama2-7b-ngram/vsd=${num_speculative_tokens}_bz=${batch_size}_input-len=${input_len}.json
    done
done