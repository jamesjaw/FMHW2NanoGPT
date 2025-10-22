#!/usr/bin/env python3
"""
測試模型從英文到中文的轉變過程
在 finetune 過程中定期運行此腳本來觀察變化
"""

import torch
from model import GPTConfig, GPT
import pickle
import os

def load_model(checkpoint_path):
    """載入模型"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_meta(data_dir):
    """載入字符映射"""
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    return meta['stoi'], meta['itos']

def generate_text(model, stoi, itos, prompt="", max_tokens=100, temperature=0.8):
    """生成文本"""
    model.eval()
    
    # 編碼 prompt
    if prompt:
        prompt_ids = [stoi.get(c, 0) for c in prompt]
    else:
        prompt_ids = [stoi.get('\n', 0)]  # 從換行開始
    
    x = torch.tensor(prompt_ids, dtype=torch.long)[None, ...]
    
    # 生成
    with torch.no_grad():
        for _ in range(max_tokens):
            logits, _ = model(x)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)
    
    # 解碼
    generated_ids = x[0].tolist()
    generated_text = ''.join([itos[i] for i in generated_ids])
    return generated_text

def test_model_transition(checkpoint_path, data_dir='data/chinese_char'):
    """測試模型的轉變狀態"""
    print(f"\n=== 測試模型: {checkpoint_path} ===")
    
    # 載入模型和字符映射
    model = load_model(checkpoint_path)
    stoi, itos = load_meta(data_dir)
    
    # 測試不同的 prompt
    test_prompts = [
        "",  # 無 prompt
        "To be or not to be",  # 英文經典
        "Once upon a time",    # 英文故事開頭
        "The quick brown",     # 英文常見句子
    ]
    
    print("生成結果:")
    for i, prompt in enumerate(test_prompts):
        print(f"\nPrompt {i+1}: '{prompt}'")
        generated = generate_text(model, stoi, itos, prompt, max_tokens=50)
        print(f"Generated: {generated[:100]}...")
        
        # 簡單分析中英文比例
        chinese_chars = sum(1 for c in generated if '\u4e00' <= c <= '\u9fff')
        english_chars = sum(1 for c in generated if c.isalpha())
        total_chars = len(generated)
        
        if total_chars > 0:
            chinese_ratio = chinese_chars / total_chars * 100
            english_ratio = english_chars / total_chars * 100
            print(f"中文比例: {chinese_ratio:.1f}%, 英文比例: {english_ratio:.1f}%")

if __name__ == "__main__":
    # 測試原始 shakespeare 模型
    print("=== 原始 Shakespeare 模型 ===")
    test_model_transition('out-shakespeare-char/ckpt.pt')
    
    # 測試 finetune 後的模型 (如果存在)
    gentle_path = 'out-chinese-gentle/ckpt.pt'
    if os.path.exists(gentle_path):
        print("\n=== Finetune 後的模型 ===")
        test_model_transition(gentle_path)
    else:
        print(f"\nFinetune 模型尚未生成: {gentle_path}")
        print("請先運行: python train.py config/finetune_chinese_gentle.py --device=mps --compile=False")