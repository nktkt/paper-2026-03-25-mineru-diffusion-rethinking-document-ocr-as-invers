"""Unified tokenizer for MinerU-Diffusion.

The paper uses a shared vocabulary covering:
- Text tokens (characters, subwords)
- Layout markers (<|box_start|>, <|box_end|>, <|ref_start|>, etc.)
- Table delimiters (<fcel>, <nl>, etc. in OTSL format)
- Math operators and LaTeX symbols
- Special tokens ([MASK], [PAD], [BOS], [EOS])

This module provides a simple character-level tokenizer with special tokens
for local experiments, plus utilities for integrating with HuggingFace tokenizers.
"""

from typing import Dict, List, Optional, Tuple


# Special tokens
SPECIAL_TOKENS = {
    "[MASK]": 0,
    "[PAD]": 1,
    "[BOS]": 2,
    "[EOS]": 3,
    "[UNK]": 4,
}

# Layout tokens from the paper
LAYOUT_TOKENS = {
    "<|box_start|>": 5,
    "<|box_end|>": 6,
    "<|ref_start|>": 7,
    "<|ref_end|>": 8,
    "<|rotate_up|>": 9,
    "<|rotate_right|>": 10,
    "<|rotate_down|>": 11,
    "<|rotate_left|>": 12,
}

# Table tokens (OTSL format)
TABLE_TOKENS = {
    "<fcel>": 13,
    "<ecel>": 14,
    "<nl>": 15,
    "<lcel>": 16,
    "<ucel>": 17,
}

RESERVED_OFFSET = 32  # Character tokens start at this offset


class OCRTokenizer:
    """Character-level tokenizer with OCR-specific special tokens.

    Supports encoding/decoding text with layout markers, table delimiters,
    and special diffusion tokens ([MASK], [PAD], etc.).
    """

    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self.special_tokens = {**SPECIAL_TOKENS, **LAYOUT_TOKENS, **TABLE_TOKENS}
        self.id_to_special = {v: k for k, v in self.special_tokens.items()}

        # Character mapping (starts at RESERVED_OFFSET)
        self._char_to_id: Dict[str, int] = {}
        self._id_to_char: Dict[int, str] = {}
        self._next_id = RESERVED_OFFSET

        # Pre-populate with printable ASCII
        for i in range(32, 127):
            ch = chr(i)
            self._register_char(ch)

    def _register_char(self, char: str) -> int:
        if char in self._char_to_id:
            return self._char_to_id[char]
        if self._next_id >= self.vocab_size:
            return self.special_tokens["[UNK]"]
        idx = self._next_id
        self._char_to_id[char] = idx
        self._id_to_char[idx] = char
        self._next_id += 1
        return idx

    @property
    def mask_token_id(self) -> int:
        return SPECIAL_TOKENS["[MASK]"]

    @property
    def pad_token_id(self) -> int:
        return SPECIAL_TOKENS["[PAD]"]

    @property
    def bos_token_id(self) -> int:
        return SPECIAL_TOKENS["[BOS]"]

    @property
    def eos_token_id(self) -> int:
        return SPECIAL_TOKENS["[EOS]"]

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """Encode text into token IDs.

        Handles special token patterns like <|box_start|> and <fcel>.

        Args:
            text: Input text (may contain special token markers).
            add_special_tokens: Whether to prepend [BOS] and append [EOS].
            max_length: Truncate to this length (including special tokens).

        Returns:
            List of token IDs.
        """
        ids = []
        if add_special_tokens:
            ids.append(self.bos_token_id)

        i = 0
        while i < len(text):
            matched = False
            # Try to match special tokens (longest match first)
            for token, token_id in sorted(
                self.special_tokens.items(), key=lambda x: -len(x[0])
            ):
                if text[i:].startswith(token):
                    ids.append(token_id)
                    i += len(token)
                    matched = True
                    break
            if not matched:
                char = text[i]
                if char in self._char_to_id:
                    ids.append(self._char_to_id[char])
                else:
                    idx = self._register_char(char)
                    ids.append(idx)
                i += 1

        if add_special_tokens:
            ids.append(self.eos_token_id)

        if max_length is not None:
            ids = ids[:max_length]

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text.

        Args:
            ids: Token IDs to decode.
            skip_special_tokens: Whether to skip [BOS], [EOS], [PAD], [MASK].

        Returns:
            Decoded text string.
        """
        skip_ids = set()
        if skip_special_tokens:
            skip_ids = {
                SPECIAL_TOKENS["[MASK]"],
                SPECIAL_TOKENS["[PAD]"],
                SPECIAL_TOKENS["[BOS]"],
                SPECIAL_TOKENS["[EOS]"],
            }

        parts = []
        for idx in ids:
            if idx in skip_ids:
                continue
            if idx in self.id_to_special:
                parts.append(self.id_to_special[idx])
            elif idx in self._id_to_char:
                parts.append(self._id_to_char[idx])
            else:
                parts.append("[UNK]")

        return "".join(parts)

    def batch_encode(
        self,
        texts: List[str],
        max_length: int = 512,
        padding: bool = True,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """Encode a batch of texts with padding.

        Args:
            texts: List of input texts.
            max_length: Maximum sequence length.
            padding: Whether to pad to max_length.

        Returns:
            Tuple of (input_ids, attention_mask).
        """
        all_ids = [self.encode(text, max_length=max_length) for text in texts]

        if padding:
            max_len = min(max(len(ids) for ids in all_ids), max_length)
            attention_masks = []
            for i in range(len(all_ids)):
                cur_len = len(all_ids[i])
                mask = [1] * cur_len + [0] * (max_len - cur_len)
                all_ids[i] = all_ids[i] + [self.pad_token_id] * (max_len - cur_len)
                attention_masks.append(mask)
        else:
            attention_masks = [[1] * len(ids) for ids in all_ids]

        return all_ids, attention_masks

    def __len__(self) -> int:
        return self._next_id
