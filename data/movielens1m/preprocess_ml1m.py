import pandas as pd, numpy as np, os, json
from tqdm import tqdm

def preprocess_ml(path_in="ratings.dat", path_out="data/ml1m/proc", min_user_len=5, min_item_freq=5, rating_th=3.0):
    # 1. 读原始文件
    df = pd.read_csv(path_in, sep="::", names=["userId", "movieId", "rating", "timestamp"], engine="python")
    print("原始记录数:", len(df))

    # 2. 过滤评分
    df = df[df["rating"] >= rating_th]

    # 3. 按时间排序
    df = df.sort_values(["userId", "timestamp"])

    # 4. 过滤低频物品和短序列用户
    item_freq = df["movieId"].value_counts()
    keep_items = set(item_freq[item_freq >= min_item_freq].index)
    df = df[df["movieId"].isin(keep_items)]

    user_len = df.groupby("userId").size()
    keep_users = set(user_len[user_len >= min_user_len].index)
    df = df[df["userId"].isin(keep_users)]

    print(f"过滤后: 用户 {df['userId'].nunique()} | 物品 {df['movieId'].nunique()} | 记录 {len(df)}")

    # 5. 建立映射 (从1开始；0保留作PAD)
    user2idx = {int(u): int(i+1) for i, u in enumerate(sorted(df["userId"].unique()))}
    item2idx = {int(v): int(i+1) for i, v in enumerate(sorted(df["movieId"].unique()))}

    df["uidx"] = df["userId"].map(user2idx)
    df["iidx"] = df["movieId"].map(item2idx)

    # 6. 转成序列格式
    seqs = df.groupby("uidx")["iidx"].apply(list).to_dict()
    print("平均序列长度:", np.mean([len(v) for v in seqs.values()]))

    # 7. 保存映射与序列
    os.makedirs(path_out, exist_ok=True)
    with open(os.path.join(path_out, "user2idx.json"), "w") as f: json.dump(user2idx, f)
    with open(os.path.join(path_out, "item2idx.json"), "w") as f: json.dump(item2idx, f)

    # 写成一个 .txt：每行 userId item1 item2 ... itemN
    with open(os.path.join(path_out, "data.txt"), "w") as f:
        for u, items in seqs.items():
            f.write(" ".join(map(str, [u] + items)) + "\n")
    print(f"保存到 {path_out}/data.txt ✅")

if __name__ == "__main__":
    path = os.path.dirname(__file__)
    path_in = os.path.join(path, 'ratings.dat')
    path_out = os.path.join(path, 'proc')
    # print(path)
    preprocess_ml(path_in, path_out)
