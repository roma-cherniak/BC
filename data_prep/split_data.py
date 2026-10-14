import glob
import random
from data_colection.lidar_collection import DATA_DIR

files = sorted(glob.glob(f"{DATA_DIR}/drive_*.npz"))
random.seed(42)
random.shuffle(files)

n = len(files)

train =  files[:int(n * 0.8)]
val = files[int(n * 0.8): int(n * 0.9)]
test = files[int(n * 0.9): int(n * 1)]


print(f"Len of train file - {len(train)} \n Len of val file - {len(val)} \n Len of test file - {len(test)}")



for name, part in [("train", train), ("val", val), ("test", test)]:
    with open(f"data/{name}.txt", "w") as f:
        f.write("\n".join(part))


