python3 bench.py \
--model /data/yutao/qwen3/Qwen3-1.7B-Base/ \
--dtype float16 \
--max_model_len 32768 \
--max_num_batched_tokens 131072 \
--tensor_parallel_size 1 \
--gpu_memory_utilization 0.9 \
--sparse_decoding \
# --enforce_eager

