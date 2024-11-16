TARGET="lmsys/vicuna-7b-v1.5"
DRAFT="eqhylxx/vicuna-160m"
input_len=256
WARMUP=5
REPEAT=10

for batch_size in 1 2 4 8 16 32 64 128
do
    python benchmarks/benchmark_latency.py \
        --model  $TARGET \
        --disable-async-output-proc \
        --max-num-seqs 256 \
        --input-len $input_len \
        --output-len 256 \
        --batch-size $batch_size \
        --num-iters-warmup $WARMUP \
        --num-iters $REPEAT \
        --output-json benchmarks/dsd/results/llama2-7b-draft/org_bz=${batch_size}_input-len=${input_len}.json
done


for acc in 0.5
do
    for batch_size in 16
    do
        python benchmarks/benchmark_latency.py \
            --model  $TARGET \
            --max-num-seqs 256 \
            --input-len $input_len \
            --output-len 256 \
            --batch-size $batch_size \
            --num-iters-warmup $WARMUP \
            --num-iters $REPEAT \
            --speculative-model $DRAFT \
            --num-speculative-tokens 8 \
            --dsd \
            --acceptance-rate $acc \
            --output-json benchmarks/dsd/results/llama2-7b-draft/dsd_bz=${batch_size}_input-len=${input_len}_acc=${acc}.json
    done


    for batch_size in 1 2 4 8 16 32 64 128
    do
        for num_speculative_tokens in 1 3 5 7
        do
            python benchmarks/benchmark_latency.py \
                --model  $TARGET \
                --input-len $input_len \
                --output-len 256 \
                --batch-size $batch_size \
                --num-iters-warmup $WARMUP \
                --num-iters $REPEAT \
                --speculative-model $DRAFT \
                --num-speculative-tokens $num_speculative_tokens \
                --acceptance-rate $acc \
                --output-json benchmarks/dsd/results/llama2-7b-draft/vsd=${num_speculative_tokens}_bz=${batch_size}_input-len=${input_len}_acc=${acc}.json
        done
    done
done