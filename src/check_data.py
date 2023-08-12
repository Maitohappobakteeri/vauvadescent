import numpy as np
import os
import common

topics = common.list_all_files(os.path.join(common.cache_dir, "posts"))
lengths = []
for t in topics:
    data = common.load_json_file(t)
    lengths.append(len(data))

print("Tokens in data:", sum(lengths))
print("Available unique training segments:", sum(lengths) // 500)
print(f"With 32 batches this takes {(sum(lengths) // 500) // 32} iterations")
