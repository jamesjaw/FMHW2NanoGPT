def distinct_n(texts, n=2):
    all_ngrams = []
    for text in texts:
        tokens = list(text)
        #.strip().split()
        if len(tokens) >= n:  # 確保有足夠的 tokens 來生成 n-grams
            ngrams = list(zip(*[tokens[i:] for i in range(n)]))
            all_ngrams.extend(ngrams)
    unique = len(set(all_ngrams))
    total = len(all_ngrams)
    return unique / total if total > 0 else 0


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


def calculate_diversity_metrics(filename):
    """計算文本多樣性指標"""
    texts = read_file_line_by_line(filename)
    
    if not texts:
        print("沒有有效的文本數據")
        return None, None
    
    d1 = distinct_n(texts, n=1)
    d2 = distinct_n(texts, n=2)
    
    print(f"檔案: {filename}")
    print(f"總行數: {len(texts)}")
    print(f"Distinct-1: {d1:.3f}")
    print(f"Distinct-2: {d2:.3f}")
    
    return d1, d2


# 使用範例
if __name__ == "__main__":
    # 替換成你的檔案路徑
    filename = "output3.txt"
    d1, d2 = calculate_diversity_metrics(filename)
