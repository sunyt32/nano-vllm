python3 bench.py \
--model /data/yutao/qwen3/Qwen3-8B-Base/ \
--dtype bfloat16 \
--max_model_len 131072 \
--max_num_batched_tokens 131072 \
--tensor_parallel_size 1 \
--gpu_memory_utilization 0.8 \
--sparse_decoding \
# --enforce_eager

