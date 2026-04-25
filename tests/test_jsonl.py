from src.utils.json_io import iter_jsonl_batches

batch = iter_jsonl_batches("data/index/full.jsonl", batch_size=5)

print(next(batch)[1])