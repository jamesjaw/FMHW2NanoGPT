import numpy as np
from scipy.spatial.distance import jensenshannon
from collections import Counter


def js_divergence(train_texts, gen_texts):
    """計算 Jensen-Shannon Divergence"""
    
    train_tokens = []
    gen_tokens = []
    
    for text in train_texts:
        train_tokens.extend(list(text))
    
    for text in gen_texts:
        gen_tokens.extend(list(text))
    
    train_counts = Counter(train_tokens)
    gen_counts = Counter(gen_tokens)

    vocab = list(set(train_counts) | set(gen_counts))
    p = np.array([train_counts[w] for w in vocab], dtype=float)
    q = np.array([gen_counts[w] for w in vocab], dtype=float)

    p /= p.sum()
    q /= q.sum()

    return jensenshannon(p, q)  # the small the better


def read_file_line_by_line(filename):
    """逐行讀取檔案並返回文本列表"""
    texts = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:  # 跳過空行
                    texts.append(line)
        print(f"成功讀取 {len(texts)} 行文本")
        return texts
    except FileNotFoundError:
        print(f"檔案 {filename} 不存在")
        return []
    except Exception as e:
        print(f"讀取檔案時發生錯誤: {e}")
        return []


def calculate_js_divergence(train_file, gen_file):
    """計算兩個檔案之間的 JS Divergence"""
    train_texts = read_file_line_by_line(train_file)
    gen_texts = read_file_line_by_line(gen_file)
    
    if not train_texts or not gen_texts:
        print("無法讀取有效的文本數據")
        return None
    
    js_score = js_divergence(train_texts, gen_texts)
    
    print(f"訓練檔案: {train_file} ({len(train_texts)} 行)")
    print(f"生成檔案: {gen_file} ({len(gen_texts)} 行)")
    print(f"Jensen-Shannon Divergence: {js_score:.4f}")
    
    return js_score



if __name__ == "__main__":
    
    js_score = calculate_js_divergence("input3.txt", "output3.txt")
