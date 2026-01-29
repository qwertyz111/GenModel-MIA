
# ag_news
accelerate launch ./ft_llms/llms_finetune.py \
--output_dir ./ft_llms/llama/ag_news/target/ \
--block_size 128 --eval_steps 100 --save_epochs 100 --log_steps 100 \
-d ag_news -m decapoda-research/llama-7b-hf --packing --use_dataset_cache \
-e 10 -b 4 -lr 1e-4 --gradient_accumulation_steps 1 \
--train_sta_idx=0 --train_end_idx=10000 --eval_sta_idx=0 --eval_end_idx=1000

# refer candidate
accelerate launch ./ft_llms/llms_finetune.py \
--output_dir ./ft_llms/llama/ag_news/candidate/ \
--block_size 128 --eval_steps 100 --save_epochs 100 --log_steps 100 \
-d JulesBelveze/tldr_news -m decapoda-research/llama-7b-hf --packing --use_dataset_cache \
-e 10 -b 4 -lr 1e-4 --gradient_accumulation_steps 1 \
--train_sta_idx=0 --train_end_idx=4767 --eval_sta_idx=0 --eval_end_idx=538

# refer oracle
accelerate launch ./ft_llms/llms_finetune.py \
--output_dir ./ft_llms/llama/ag_news/oracle/ \
--block_size 128 --eval_steps 100 --save_epochs 100 --log_steps 100 \
-d ag_news -m decapoda-research/llama-7b-hf --packing --use_dataset_cache \
-e 10 -b 4 -lr 1e-4 --gradient_accumulation_steps 1 \
--train_sta_idx=10000 --train_end_idx=20000 --eval_sta_idx=1000 --eval_end_idx=2000

accelerate launch refer_data_generate.py \
-tm /mnt/data0/fuwenjie/MIA-LLMs/ft_llms/llama/ag_news/target/checkpoint-3000 \
-m decapoda-research/llama-7b-hf -d ag_news

# refer prompt
accelerate launch ./ft_llms/llms_finetune.py --refer \
--output_dir ./ft_llms/llama/ag_news/refer/ \
--block_size 128 --eval_steps 100 --save_epochs 100 --log_steps 100 \
-d ag_news -m decapoda-research/llama-7b-hf --packing --use_dataset_cache \
-e 2 -b 4 -lr 5e-5 --gradient_accumulation_steps 1 \
--train_sta_idx=0 --train_end_idx=10000 --eval_sta_idx=0 --eval_end_idx=1000