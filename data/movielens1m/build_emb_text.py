# tools/build_items_text.py
import os, csv, argparse, json

def build_text_ml1m(movies_dat_path, out_txt_path):
    """
    输入: movies.dat (MovieID::Title::Genres)
    输出: items.txt (item_id \t text)
    """
    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)
    with open(movies_dat_path, 'r', encoding='latin-1') as f, \
         open(out_txt_path, 'w', encoding='utf-8') as w:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                mid, title, genres = line.split("::")
            except ValueError:
                # 有些语料不是双冒号分隔，请按需要修改
                parts=line.split("::")
                mid, title, genres = parts[0], parts[1], parts[-1]
            text = f"title: {title} ; genres: {genres}"
            w.write(f"{mid}\t{text}\n")

if __name__ == "__main__":
    
    path = os.path.dirname(__file__)
    path_in = os.path.join(path, 'movies.dat')
    path_out = os.path.join(path, 'proc', 'text.txt')
    
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--movies_dat", required=True)
    # ap.add_argument("--out_txt", required=True)
    # args = ap.parse_args()

    build_text_ml1m(path_in, path_out)
    print("Wrote", path_out)
