"""Tests for OCR tokenizer."""

import pytest
from mineru_diffusion.tokenizer import OCRTokenizer, SPECIAL_TOKENS


class TestOCRTokenizer:
    def test_encode_decode_roundtrip(self):
        tok = OCRTokenizer()
        text = "Hello World"
        ids = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_special_tokens_in_output(self):
        tok = OCRTokenizer()
        ids = tok.encode("Hi", add_special_tokens=True)
        assert ids[0] == tok.bos_token_id
        assert ids[-1] == tok.eos_token_id

    def test_skip_special_tokens(self):
        tok = OCRTokenizer()
        ids = tok.encode("Hi", add_special_tokens=True)
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert decoded == "Hi"

    def test_layout_tokens(self):
        tok = OCRTokenizer()
        text = "<|box_start|>100 200 300 400<|box_end|>"
        ids = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_table_tokens(self):
        tok = OCRTokenizer()
        text = "<fcel>Name<fcel>Score<nl>"
        ids = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_max_length(self):
        tok = OCRTokenizer()
        ids = tok.encode("A" * 100, max_length=10)
        assert len(ids) <= 10

    def test_batch_encode(self):
        tok = OCRTokenizer()
        texts = ["Hello", "Hi there, world!"]
        ids, masks = tok.batch_encode(texts, max_length=20)
        assert len(ids) == 2
        assert len(ids[0]) == len(ids[1])  # Padded to same length

    def test_mask_token_id(self):
        tok = OCRTokenizer()
        assert tok.mask_token_id == 0

    def test_unicode(self):
        tok = OCRTokenizer(vocab_size=16384)
        text = "数式: α + β = γ"
        ids = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(ids)
        assert decoded == text
