import json
import os
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

CHAR_VOCAB = None


def build_char_vocab(texts: List[str]) -> dict:
    chars = set()
    for t in texts:
        chars.update(t)
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    for c in sorted(chars):
        if c not in vocab:
            vocab[c] = len(vocab)
    return vocab


def load_jsonl(path: str, max_samples: Optional[int] = None) -> List[str]:
    data = []
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            if "text" in obj:
                data.append(obj["text"])
            elif "ciphertext" in obj:
                data.append(obj["ciphertext"])
            if max_samples is not None and len(data) >= max_samples:
                break
    return data


def load_jsonl_with_ids(path: str) -> Tuple[List[str], List[str]]:
    ids, texts = [], []
    if not os.path.exists(path):
        return ids, texts
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line.strip())
            sample_id = obj.get("id", str(i))
            ids.append(sample_id)
            if "ciphertext" in obj:
                texts.append(obj["ciphertext"])
            elif "text" in obj:
                texts.append(obj["text"])
            else:
                texts.append("")
    return ids, texts


class CharTokenizer:
    def __init__(self, vocab: dict):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.pad_id = vocab["<pad>"]
        self.bos_id = vocab["<bos>"]
        self.eos_id = vocab["<eos>"]
        self.unk_id = vocab["<unk>"]
        self.vocab_size = len(vocab)

    def encode(self, text: str, add_special_tokens: bool = True, max_length: int = None) -> List[int]:
        ids = []
        if add_special_tokens:
            ids.append(self.bos_id)
        for c in text:
            ids.append(self.vocab.get(c, self.unk_id))
        if add_special_tokens:
            ids.append(self.eos_id)
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length - 1] + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        specials = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        chars = []
        for i in ids:
            if i == self.eos_id:
                break
            if skip_special_tokens and i in specials:
                continue
            chars.append(self.inv_vocab.get(i, "<unk>"))
        return "".join(chars)

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab)


class CipherDataset(Dataset):
    def __init__(
        self,
        cipher_texts: List[str],
        plain_texts: List[str],
        cipher_tokenizer: CharTokenizer,
        plain_tokenizer: CharTokenizer,
        max_src_len: int = 512,
        max_tgt_len: int = 512,
    ):
        self.cipher_tokenizer = cipher_tokenizer
        self.plain_tokenizer = plain_tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        if not plain_texts or len(plain_texts) != len(cipher_texts):
            plain_texts = ["" for _ in range(len(cipher_texts))]
        self.samples = list(zip(cipher_texts, plain_texts))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cipher, plain = self.samples[idx]
        src = self.cipher_tokenizer.encode(cipher, add_special_tokens=True, max_length=self.max_src_len)
        tgt = self.plain_tokenizer.encode(plain, add_special_tokens=True, max_length=self.max_tgt_len)
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def collate_fn(batch):
    srcs, tgts = zip(*batch)
    srcs_padded = pad_sequence(srcs, batch_first=True, padding_value=PAD_IDX)
    tgts_padded = pad_sequence(tgts, batch_first=True, padding_value=PAD_IDX)
    return srcs_padded, tgts_padded


def resolve_split_paths(data_dir: str, split: str) -> Tuple[str, str]:
    cand_pairs = [
        (os.path.join(data_dir, split, "cipher.jsonl"), os.path.join(data_dir, split, "cleaned.jsonl")),
        (os.path.join(data_dir, f"{split}_cipher.jsonl"), os.path.join(data_dir, f"{split}_cleaned.jsonl")),
        (os.path.join(data_dir, "cipher.jsonl"), os.path.join(data_dir, "cleaned.jsonl")),
    ]
    for c_path, p_path in cand_pairs:
        if os.path.exists(c_path):
            return c_path, p_path
    return cand_pairs[0]


def get_dataloader(
    split: str,
    data_dir: str,
    cipher_tokenizer: CharTokenizer,
    plain_tokenizer: CharTokenizer,
    batch_size: int = 32,
    max_src_len: int = 512,
    max_tgt_len: int = 512,
    shuffle: bool = None,
    num_workers: int = 2,
    max_samples: Optional[int] = None,
) -> DataLoader:
    cipher_path, plain_path = resolve_split_paths(data_dir, split)
    cipher_texts = load_jsonl(cipher_path, max_samples=max_samples)
    plain_texts = load_jsonl(plain_path, max_samples=max_samples) if os.path.exists(plain_path) else ["" for _ in range(len(cipher_texts))]
    dataset = CipherDataset(
        cipher_texts, plain_texts,
        cipher_tokenizer, plain_tokenizer,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
    )
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_tokenizers(data_dir: str, save_dir: str) -> Tuple[CharTokenizer, CharTokenizer]:
    c_path, p_path = resolve_split_paths(data_dir, "train")
    train_cipher = load_jsonl(c_path)
    train_plain = load_jsonl(p_path)
    cipher_vocab = build_char_vocab(train_cipher)
    plain_vocab = build_char_vocab(train_plain)
    cipher_tok = CharTokenizer(cipher_vocab)
    plain_tok = CharTokenizer(plain_vocab)
    os.makedirs(save_dir, exist_ok=True)
    cipher_tok.save(os.path.join(save_dir, "cipher_vocab.json"))
    plain_tok.save(os.path.join(save_dir, "plain_vocab.json"))
    return cipher_tok, plain_tok
