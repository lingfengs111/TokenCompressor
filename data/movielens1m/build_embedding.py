# tools/build_item_table_from_text.py
import os, argparse, json, numpy as np
import torch
from tqdm import tqdm

def load_items_list(items_txt_path, to_lower=True):
    """
    读取 items.txt: 每行  item_id \t text
    返回: [(item_id(int/str), text(str)), ...]
    """
    items=[]
    with open(items_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.rstrip("\n")
            if not line: continue
            try:
                iid, txt = line.split("\t", 1)
            except ValueError:
                parts=line.split("\t")
                iid, txt = parts[0], parts[-1]
            if to_lower: txt = txt.lower()
            items.append((iid, txt))
    return items

def encode_texts(items, model_name, batch_size, max_length, normalize):
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_length

    texts = [t for _, t in items]
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="encode"):
        batch = texts[i:i+batch_size]
        with torch.inference_mode():
            e = model.encode(batch, batch_size=len(batch), convert_to_numpy=True, device=device, normalize_embeddings=False)
        embs.append(e)
    Z = np.concatenate(embs, axis=0)  # [N, d_txt]
    if normalize:
        # L2 归一化到单位范数（可让后续 PCA 更稳）
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return Z  # float32

def fit_or_load_pca(Z, out_dim, save_pca=None, load_if_exists=True, batch_size=4096):
    """
    返回: Z_proj [N, out_dim], 以及 (mean, components) 供复用
    """
    if Z.shape[1] == out_dim:
        return Z, None
    os.makedirs(os.path.dirname(save_pca), exist_ok=True) if save_pca else None

    if save_pca and load_if_exists and os.path.exists(save_pca):
        data = np.load(save_pca)
        mean = data["mean"]         # [d_txt]
        comps = data["components"]  # [out_dim, d_txt]
        Zc = Z - mean
        Zp = Zc @ comps.T
        return Zp.astype(np.float32), (mean, comps)

    # 增量 PCA（节省内存）
    from sklearn.decomposition import IncrementalPCA
    d_txt = Z.shape[1]
    ipca = IncrementalPCA(n_components=out_dim, batch_size=batch_size)
    # 第一遍拟合增量
    for i in tqdm(range(0, Z.shape[0], batch_size), desc="fitIPCA"):
        ipca.partial_fit(Z[i:i+batch_size])
    # 变换
    Zp_list=[]
    for i in tqdm(range(0, Z.shape[0], batch_size), desc="transformIPCA"):
        Zp_list.append(ipca.transform(Z[i:i+batch_size]).astype(np.float32))
    Zp = np.concatenate(Zp_list, axis=0)
    mean = ipca.mean_.astype(np.float32)           # [d_txt]
    comps = ipca.components_.astype(np.float32)    # [out_dim, d_txt]
    if save_pca:
        np.savez_compressed(save_pca, mean=mean, components=comps)
    return Zp, (mean, comps)

def build_E_tensor(Z_proj, item2idx_path, d_model, save_path, items_txt_path=None):
    """
    Z_proj: [N, d_model] 与 items 顺序一致
    item2idx.json: {"<raw_item_id>": idx, ...}  idx ∈ [1..|I|]（0保留PAD）
    """
    with open(item2idx_path, "r") as f:
        item2idx = json.load(f)

    num_items = max(int(v) for v in item2idx.values())
    E = torch.zeros(num_items+1, d_model, dtype=torch.float32)  # 0行为PAD

    # 将 Z_proj 写入 E 的相应行
    # 注意：items_txt 的顺序可能与 item2idx 无关，所以用 dict 对齐
    # 先构建一个 map：raw_id -> row in Z_proj
    # 假设 items 的顺序与 Z_proj 的行对应
    # 读取 items_txt 以拿到 raw_id 顺序
    # 为避免重复读取，这里让调用者把 items 传进来也行；为简单起见再读一次
    raw_ids = []
    with open(items_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.rstrip("\n")
            if not line: continue
            iid = line.split("\t",1)[0]
            raw_ids.append(iid)

    assert len(raw_ids) == Z_proj.shape[0]
    for row, raw_id in enumerate(raw_ids):
        if raw_id in item2idx:
            idx = int(item2idx[raw_id])
            E[idx] = torch.from_numpy(Z_proj[row])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(E, save_path)
    return E

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # ap.add_argument("--items_txt_path", required=True)          # item_id \t text
    # ap.add_argument("--item2idx_path", required=True)           # JSON: raw -> idx (1..|I|)
    ap.add_argument("--model", default="sentence-transformers/sentence-t5-base")  # 768d，也可用 "all-MiniLM-L6-v2" 384d 等
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--normalize", default="true")
    ap.add_argument("--to_lower", default="false")
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--pca_to_d_model", default="true")
    ap.add_argument("--pca_batch_size", type=int, default=4096)
    # ap.add_argument("--save_path", default="artifacts/e_text_init.pt")
    # ap.add_argument("--save_pca", default="artifacts/pca_proj.npz")
    args = ap.parse_args()
    
    path = os.path.dirname(__file__)
    items_txt_path = os.path.join(path, 'proc', 'text.txt')
    item2idx_path = os.path.join(path, 'proc', 'item2idx.json')

    save_path = os.path.join(path, 'emb', 'e_text_init.pt')
    save_pca = os.path.join(path, 'emb', 'pca_proj.npz')
    items = load_items_list(items_txt_path, to_lower=args.to_lower)
    Z = encode_texts(items, args.model, args.batch_size, args.max_length, normalize=args.normalize)  # [N, d_txt]

    if args.pca_to_d_model and Z.shape[1] != args.d_model:
        Zp, _ = fit_or_load_pca(Z, args.d_model, save_pca=save_pca, load_if_exists=True, batch_size=args.pca_batch_size)
    else:
        Zp = Z.astype(np.float32)
        if Zp.shape[1] != args.d_model:
            raise ValueError(f"Text dim={Zp.shape[1]} != d_model={args.d_model}; enable --pca_to_d_model or choose a 512d text model.")

    E = build_E_tensor(Zp, item2idx_path, args.d_model, save_path, items_txt_path=items_txt_path)
    print("Saved:", save_path, "| shape:", tuple(E.shape))
