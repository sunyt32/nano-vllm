python3 eval_mtnd.py \
--model /data/yutao/qwen3/Qwen3-1.7B-Base/ \
--dtype bfloat16 \
--max_model_len 131072 \
--max_num_batched_tokens 131072 \
--tensor_parallel_size 1 \
--gpu_memory_utilization 0.9 \
--checkpoint /mnt/msranlp/yutao/exp/Qwen-YOCO/Qwen3-1.7B-YOCO-ALiBi-Cross128/updates_5000 \
--sparse_decoding \

