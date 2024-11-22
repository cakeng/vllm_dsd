for batch_size in 1 2 4 8 16 32 48 64 80 96 112 128 144 160 176 192 208 224 240 256
do
    for input_len in 1 512 1024 1536 2048 2560 3072 3584 4096
    do 
        python benchmarks/benchmark_latency.py \
            --model  meta-llama/Meta-Llama-3-8B-Instruct \
            --input-len $input_len \
            --output-len 10 \
            --batch-size $batch_size \
            --num-iters-warmup 5 \
            --num-iters 10 \
            --output-json benchmarks/dsd/results/llama3-8b/bz=${batch_size}_input-len=${input_len}.json
    done
done

# meta-llama/Meta-Llama-3-8B-Instruct


python benchmarks/benchmark_latency.py \
        --model lmsys/vicuna-7b-v1.5 \
        --speculative-model eqhylxx/vicuna-160m \
        --num-speculative-tokens 7 \
        --input-len 128 \
        --output-len 10 \
        --batch-size 1 \
        --num-iters-warmup 5 \
        --num-iters 10 \
        --acceptance-rate 0.7 \
        --dsd


python benchmarks/benchmark_latency.py \
        --model lmsys/vicuna-7b-v1.5 \
        --speculative-model "[ngram]" \
        --num-speculative-tokens 3 \
        --input-len 256 \
        --output-len 256 \
        --batch-size 64 \
        --num-iters-warmup 5 \
        --num-iters 10 \
        --acceptance-rate 0.9 \
        --dummy-match 0.5 \
        --ngram-prompt-lookup-max 8 \
        --ngram-prompt-lookup-min 2 \
        --max-num-batched-tokens 2048 



python benchmarks/benchmark_latency.py \
        --model lmsys/vicuna-7b-v1.5 \
        --input-len 128 \
        --output-len 10 \
        --batch-size 1 \
        --num-iters-warmup 5 \
        --num-iters 10 

turboderp/Qwama-0.5B-Instruct
# meta-llama/Llama-3.2-1B-Instruct

python benchmarks/benchmark_latency.py \
        --model meta-llama/Llama-3.1-70B-Instruct \
        --speculative-model meta-llama/Llama-3.2-1B-Instruct \
        -tp  4 \
        --num-speculative-tokens 7 \
        --input-len 128 \
        --output-len 10 \
        --batch-size 1 \
        --num-iters-warmup 5 \
        --num-iters 10 \
        --acceptance-rate 0.7 \
        --speculative-draft-tensor-parallel-size 1 \
        --dsd



python benchmarks/benchmark_serving.py \
        --model lmsys/vicuna-7b-v1.5 --dataset-name random \
        --ignore-eos --random-input-len 550 \
        --random-output-len 150 --request-rate 6 \
        --num-prompts 200 --port 8001