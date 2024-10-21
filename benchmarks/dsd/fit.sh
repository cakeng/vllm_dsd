for batch_size in 1 2 4 8 16 32 48 64 80 96 112 128 144 160 176 192 208 224 240 256
do
    for input_len in 1 512 1024 1536 2048 2560 3072 3584 4096
    do 
        python benchmarks/benchmark_latency.py \
            --model meta-llama/Llama-2-7b-hf \
            --input-len $input_len \
            --output-len 10 \
            --batch-size $batch_size \
            --num-iters-warmup 5 \
            --num-iters 10 \
            --output-json benchmarks/dsd/results/llama2-7b/bz=${batch_size}_input-len=${input_len}.json
    done
done

# meta-llama/Meta-Llama-3-8B-Instruct


# python benchmarks/benchmark_latency.py \
#         --model meta-llama/Llama-2-7b-hf \
#         --input-len 1 \
#         --output-len 1 \
#         --batch-size 2 \
#         --num-iters-warmup 5 \
#         --num-iters 10 