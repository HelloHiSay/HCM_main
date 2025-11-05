import os
import torch

# ✅ 正确导入方式（同级目录）
from hf_math_tokenizer import HFMathTokenizer

print("✅ 正在创建 HFMathTokenizer...")

tokenizer = HFMathTokenizer()
print(f"✅ 词表大小: {tokenizer.vocab_size}")

text = "sin(pi*x)+log(e)=1"
print(f"\n📝 测试文本: {text}")
encoded = tokenizer(text, return_tensors="pt")
print(f"🔢 Input IDs shape: {encoded.input_ids.shape}")
print(f"🔢 Input IDs: {encoded.input_ids.tolist()}")

decoded = tokenizer.decode(encoded.input_ids[0], skip_special_tokens=False)
print(f"🔤 解码结果: {repr(decoded)}")

# 保存到当前目录
save_dir = "./saved_hf_tokenizer"
os.makedirs(save_dir, exist_ok=True)
print(f"\n💾 正在保存到: {save_dir}")
tokenizer.save_pretrained(save_dir)

print("\n🎉 成功！请检查 ./saved_hf_tokenizer/ 目录下的文件。")