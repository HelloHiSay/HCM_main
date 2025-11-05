# hf_math_tokenizer.py
import json
import os
from typing import List, Optional, Union
from transformers import PreTrainedTokenizer
    
class HFMathTokenizer(PreTrainedTokenizer):
    # 必须定义的属性
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        special_tokens=None,
        operators=None,
        functions=None,
        constants=None,
        variables=None,
        **kwargs
    ):
        # 默认特殊 token
        if special_tokens is None:
            special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
        if operators is None:
            operators = ["+", "-", "*", "/", "^", "=", "(", ")", ".", ",", " "]
        if functions is None:
            functions = ["sin", "cos", "tan", "log", "ln", "exp", "sqrt", "abs"]
        if constants is None:
            constants = ["pi", "e"]
        if variables is None:
            variables = ["x", "y", "z", "a", "b", "c"]

        # 构建词表
        vocab_list = special_tokens + functions + constants + variables + operators
        digits = [str(i) for i in range(10)]
        vocab_list += digits

        # 去重
        seen = set()
        vocab_clean = []
        for t in vocab_list:
            if t not in seen:
                vocab_clean.append(t)
                seen.add(t)

        self.vocab = vocab_clean
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        bos_token = kwargs.pop("bos_token", "<s>")
        eos_token = kwargs.pop("eos_token", "</s>")
        unk_token = kwargs.pop("unk_token", "<unk>")
        pad_token = kwargs.pop("pad_token", "<pad>")
        # # 特殊 token
        # pad_token = "<pad>"
        # unk_token = "<unk>"
        # bos_token = "<s>"
        # eos_token = "</s>"

        # 调用父类初始化
        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs
        )

        # 为最长匹配准备
        self.sorted_tokens = sorted(self.token_to_id.keys(), key=lambda x: -len(x))
    # === 必须添加的方法 ===
    def get_vocab(self):
        return dict(self.token_to_id)
    
    @property
    def vocab_size(self):
        return len(self.vocab)

    def _tokenize(self, text: str) -> List[str]:
        """将文本切分为 token 字符串列表（HF 要求）"""
        tokens = []
        i = 0
        while i < len(text):
            matched = False
            for token in self.sorted_tokens:
                if text.startswith(token, i):
                    tokens.append(token)
                    i += len(token)
                    matched = True
                    break
            if not matched:
                tokens.append(self.unk_token)
                i += 1
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self.id_to_token.get(index, self.unk_token)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        """保存词表，HF 要求实现此方法"""
        if not os.path.isdir(save_directory):
            raise ValueError(f"Directory {save_directory} not found")

        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "_" if filename_prefix else "") + "vocab.json"
        )
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)
        return (vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """添加 BOS/EOS"""
        if self.bos_token_id is not None:
            token_ids_0 = [self.bos_token_id] + token_ids_0
        if self.eos_token_id is not None:
            token_ids_0 = token_ids_0 + [self.eos_token_id]
        return token_ids_0

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """用于 HF 内部，标记 special tokens 位置"""
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1,
                already_has_special_tokens=True
            )
        mask = [1]  # for
# # math_tokenizer_with_functions.py
# import json
# import torch
# from typing import List, Dict, Optional

# class MathTokenizer:
#     def __init__(
#         self,
#         vocab: Optional[List[str]] = None,
#         special_tokens: Optional[List[str]] = None,
#         operators: Optional[List[str]] = None,
#         functions: Optional[List[str]] = None,
#         constants: Optional[List[str]] = None,
#         variables: Optional[List[str]] = None,
#     ):
#         # 默认特殊 token
#         if special_tokens is None:
#             special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
#         # 默认运算符
#         if operators is None:
#             operators = ["+", "-", "*", "/", "^", "=", "(", ")", ".", ",", " "]
#         # 默认初等函数
#         if functions is None:
#             functions = ["sin", "cos", "tan", "log", "ln", "exp", "sqrt", "abs"]
#         # 默认常量
#         if constants is None:
#             constants = ["pi", "e"]
#         # 默认变量
#         if variables is None:
#             variables = ["x", "y", "z", "a", "b", "c"]

#         # 构建词表：**multi-char tokens 放前面**（最长匹配优先）
#         vocab_list = special_tokens + functions + constants + variables + operators
#         digits = [str(i) for i in range(10)]
#         vocab_list += digits

#         # 去重并保持顺序
#         seen = set()
#         vocab_clean = []
#         for t in vocab_list:
#             if t not in seen:
#                 vocab_clean.append(t)
#                 seen.add(t)

#         self.vocab = vocab_clean
#         self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(self.vocab)}
#         self.id_to_token: Dict[int, str] = {i: t for t, i in self.token_to_id.items()}

#         # 特殊 token ID
#         self.pad_id = self.token_to_id["<pad>"]
#         self.unk_id = self.token_to_id["<unk>"]
#         self.bos_id = self.token_to_id.get("<s>", None)
#         self.eos_id = self.token_to_id.get("</s>", None)

#         # 为最长匹配准备：按长度降序排序
#         self.sorted_tokens = sorted(self.token_to_id.keys(), key=lambda x: -len(x))

#     def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
#         ids = []
#         if add_special_tokens and self.bos_id is not None:
#             ids.append(self.bos_id)

#         i = 0
#         while i < len(text):
#             matched = False
#             for token in self.sorted_tokens:
#                 if text.startswith(token, i):
#                     ids.append(self.token_to_id[token])
#                     i += len(token)
#                     matched = True
#                     break
#             if not matched:
#                 # 未知字符 → <unk>
#                 ids.append(self.unk_id)
#                 i += 1

#         if add_special_tokens and self.eos_id is not None:
#             ids.append(self.eos_id)
#         return ids

#     def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
#         special_ids = {self.pad_id, self.unk_id}
#         if self.bos_id is not None:
#             special_ids.add(self.bos_id)
#         if self.eos_id is not None:
#             special_ids.add(self.eos_id)

#         tokens = []
#         for i in ids:
#             if skip_special_tokens and i in special_ids:
#                 continue
#             token = self.id_to_token.get(i, "<unk>")
#             if not (skip_special_tokens and token in ["<pad>", "<unk>", "<s>", "</s>"]):
#                 tokens.append(token)
#         return "".join(tokens)

#     @property
#     def vocab_size(self) -> int:
#         return len(self.vocab)

#     def save_vocab(self, file_path: str):
#         # 保存为 {token: id} 格式，类似 Qwen 风格
#         token_to_id_dict = {token: idx for token, idx in self.token_to_id.items()}
#         with open(file_path, "w", encoding="utf-8") as f:
#             json.dump(token_to_id_dict, f, ensure_ascii=False, indent=2)

#     @classmethod
#     def from_file(cls, file_path: str):
#         with open(file_path, "r", encoding="utf-8") as f:
#             token_to_id_dict = json.load(f)
#         vocab = sorted(token_to_id_dict, key=token_to_id_dict.get)  # 按 ID 排序还原列表
#         return cls(vocab=vocab)

#     def print_vocab_with_ids(self):
#         print("Vocabulary (token: id):")
#         for token, idx in self.token_to_id.items():
#             print(f'"{token}": {idx}')

#     def __repr__(self):
#         return f"MathTokenizer(vocab_size={self.vocab_size})"
#     def __call__(
#         self,
#         text: str,
#         max_length: int = 512,
#         padding: str = "max_length",
#         truncation: bool = True,
#         return_tensors: str = "pt",
#         add_special_tokens: bool = True,
#     ):
#         ids = self.encode(text, add_special_tokens=add_special_tokens)
    
#         if truncation and len(ids) > max_length:
#             ids = ids[:max_length]
    
#         if padding == "max_length":
#             if len(ids) < max_length:
#                 ids = ids + [self.pad_id] * (max_length - len(ids))
    
#         if return_tensors == "pt":
#             input_ids = torch.tensor([ids], dtype=torch.long)
#         else:
#             input_ids = [ids]
    
#         class Encoding:
#             def __init__(self, input_ids):
#                 self.input_ids = input_ids
    
#         return Encoding(input_ids)    

# # ========================
# # 使用示例
# # ========================
# if __name__ == "__main__":
#     tokenizer = MathTokenizer()

#     # 测试复杂表达式
#     test_expr = "sin(pi * x) + log(e) = 1\nsqrt(16) = 4, abs(-5) = 5"

#     print("原始表达式:")
#     print(repr(test_expr))
#     print("\n编码 IDs:")
#     ids = tokenizer.encode(test_expr)
#     print(ids)
#     print("\n解码结果:")
#     decoded = tokenizer.decode(ids)
#     print(repr(decoded))

#     print(f"\n词表大小: {tokenizer.vocab_size}")
#     tokenizer.print_vocab_with_ids()

#     # 保存词表（格式如 "*": 9）
#     tokenizer.save_vocab("tokenizers/math_tokenizer_vocab.json")
#     print("\n词表已保存为 'math_tokenizer_vocab.json'")

#     # 从文件加载
#     loaded = MathTokenizer.from_file("math_tokenizer_vocab.json")
#     print(f"重新加载的 tokenizer 词表大小: {loaded.vocab_size}")