from datasets import load_dataset

# dataset = load_dataset('json', data_files='/home/andoni.sudupe/LLaMA-Omni/data/instruct_en_200k_data_cosy2.json')  # example
dataset = load_dataset('Ansu/instruct_S2S_cleaned_en')  # example
n_shards = 10
dataset = dataset['train']

for i in range(n_shards):
    shard = dataset.shard(num_shards=n_shards, index=i)
    shard.save_to_disk(f"/home/andoni.sudupe/LLaMA-Omni/data/shards/shard_{i}")
