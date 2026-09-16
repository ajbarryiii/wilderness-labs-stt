"""CTC tokenization: blank is zero; every SentencePiece ID is shifted by one."""

from __future__ import annotations

import itertools
from pathlib import Path
import unicodedata
from typing import Iterable


def artifact_path(path: str | Path) -> Path:
    """Validate the mounted artifact drive before creating any model/cache files."""
    from .storage import ensure_artifact_path

    return ensure_artifact_path(path)


def normalize_text(text: str, *, casefold: bool = True) -> str:
    """NFKC and whitespace normalization; preserve digits, punctuation and negation."""
    if not isinstance(text, str):
        raise TypeError("Transcript must be a string")
    text = " ".join(unicodedata.normalize("NFKC", text).split())
    return text.casefold() if casefold else text


class SpeechTokenizer:
    blank_id = 0

    def __init__(self, path: str | Path, *, casefold: bool = True):
        import sentencepiece as spm

        self.path = Path(path)
        self.casefold = casefold
        self.processor = spm.SentencePieceProcessor(model_file=str(path))
        self.vocab_size = self.processor.get_piece_size() + 1
        self.unk_id = self.processor.unk_id() + 1

    def encode(self, text: str) -> list[int]:
        ids = self.processor.encode(normalize_text(text, casefold=self.casefold), out_type=int)
        if self.processor.unk_id() in ids:
            raise ValueError("Tokenizer emitted UNK; train with byte_fallback enabled")
        return [index + 1 for index in ids]

    def decode(self, ids: Iterable[int]) -> str:
        values = [int(index) for index in ids if int(index) != self.blank_id]
        if any(index < 1 or index >= self.vocab_size for index in values):
            raise ValueError("Token ID outside vocabulary")
        return self.processor.decode([index - 1 for index in values])

    def decode_ctc(self, ids: Iterable[int]) -> str:
        """Collapse consecutive frame IDs first, then remove CTC blanks."""
        return self.decode(index for index, _ in itertools.groupby(int(i) for i in ids))


def train_tokenizer(
    texts: Iterable[str],
    output_prefix: str | Path,
    vocab_size: int = 2048,
    *,
    max_samples: int = 1_000_000,
    max_chars: int = 100_000_000,
    max_sentence_chars: int = 4096,
    casefold: bool = True,
    seed: int = 0,
) -> SpeechTokenizer:
    """Train deterministic BPE from a bounded text stream, including 256 byte pieces.

    ``vocab_size`` includes CTC blank. No corpus copy is written. SentencePiece
    still holds its bounded training sample in RAM; reduce ``max_chars`` on
    memory constrained hosts. A too-small corpus/vocabulary fails explicitly.
    """
    import sentencepiece as spm

    if vocab_size < 260:
        raise ValueError("Byte-fallback SentencePiece requires at least 260 total outputs")
    if min(max_samples, max_chars, max_sentence_chars) <= 0:
        raise ValueError("Tokenizer sample and character budgets must be positive")
    prefix = artifact_path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if Path(str(prefix) + ".model").exists() or Path(str(prefix) + ".vocab").exists():
        raise FileExistsError(f"Refusing to overwrite existing tokenizer: {prefix}")

    def bounded_texts():
        used = 0
        for text in itertools.islice(texts, max_samples):
            sentence = normalize_text(text, casefold=casefold)
            if not sentence or len(sentence) > max_sentence_chars:
                continue
            if used + len(sentence) > max_chars:
                break
            used += len(sentence)
            yield sentence

    spm.set_random_generator_seed(seed)
    spm.SentencePieceTrainer.train(
        sentence_iterator=bounded_texts(),
        model_prefix=str(prefix),
        vocab_size=vocab_size - 1,
        model_type="bpe",
        byte_fallback=True,
        character_coverage=1.0,
        normalization_rule_name="identity",
        num_threads=1,
        shuffle_input_sentence=False,
        input_sentence_size=0,
        max_sentence_length=max_sentence_chars * 4,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0,
        hard_vocab_limit=True,
        minloglevel=2,
    )
    return SpeechTokenizer(str(prefix) + ".model", casefold=casefold)


class CharacterTokenizer:
    """Small, explicit fixture tokenizer for smoke tests; unknown text is an error."""

    blank_id = 0

    def __init__(self, alphabet: str = " abcdefghijklmnopqrstuvwxyz0123456789.,?!'-", *, casefold: bool = True):
        if len(set(alphabet)) != len(alphabet) or not alphabet:
            raise ValueError("Alphabet must be nonempty with unique characters")
        self.alphabet = alphabet
        self.casefold = casefold
        self.vocab_size = len(alphabet) + 1
        self._ids = {char: index + 1 for index, char in enumerate(alphabet)}

    def encode(self, text: str) -> list[int]:
        normalized = normalize_text(text, casefold=self.casefold)
        try:
            return [self._ids[char] for char in normalized]
        except KeyError as error:
            raise ValueError(f"Character outside fixture alphabet: {error.args[0]!r}") from error

    def decode(self, ids: Iterable[int]) -> str:
        values = [int(index) for index in ids if int(index) != self.blank_id]
        if any(index < 1 or index >= self.vocab_size for index in values):
            raise ValueError("Token ID outside vocabulary")
        return "".join(self.alphabet[index - 1] for index in values)

    def decode_ctc(self, ids: Iterable[int]) -> str:
        return self.decode(index for index, _ in itertools.groupby(int(i) for i in ids))
