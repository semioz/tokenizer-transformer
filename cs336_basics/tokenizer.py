from collections.abc import Iterable, Iterator
import os
import pickle
from cs336_basics.train_bpe import PAT
import regex as re

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = special_tokens or []

        # special tokens should be appended to the vocabulary if they aren’t already there
        existing_tokens = set(self.vocab.values())
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")

            if token_bytes not in existing_tokens:
                self.vocab[len(self.vocab)] = token_bytes
                existing_tokens.add(token_bytes)

        self.bytes_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}  #  earlier merges have higher priority

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if text == "":
            return []
        ids = []

        for part in self._split_special_tokens(text):
            if part == "":
                continue

            if part in self.special_tokens:
                ids.append(self.bytes_to_id[part.encode("utf-8")])
                continue

            for match in re.finditer(PAT, part):
                pretoken = match.group()
                ids.extend(self._encode_pretoken(pretoken))
        return ids

    def _split_special_tokens(self, text: str) -> list[str]:
        if not self.special_tokens:
            return [text]

        special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "(" + "|".join(re.escape(token) for token in special_tokens) + ")"
        return re.split(pattern, text)

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        pieces = tuple(bytes([byte]) for byte in pretoken.encode("utf-8"))

        while len(pieces) > 1:
            best_pair = None
            best_rank = float("inf")

            for i in range(len(pieces) - 1):
                pair = (pieces[i], pieces[i+1])
                rank = self.merge_ranks.get(pair, float("inf"))
                if rank < best_rank:
                    best_pair = pair
                    best_rank = rank
            if best_pair is None:
                break

            pieces = self._merge_pair(pieces, best_pair)

        return [self.bytes_to_id[piece] for piece in pieces]

    def _merge_pair(self, pieces:tuple[bytes, ...], pair:tuple[bytes, bytes]) -> tuple[bytes, ...]:
        merged = []
        i = 0

        while i < len(pieces):
            if i < len(pieces) - 1 and (pieces[i], pieces[i + 1]) == pair:
                merged.append(pieces[i] + pieces[i + 1])
                i += 2
            else:
                merged.append(pieces[i])
                i += 1

        return tuple(merged)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        decoded_bytes = b""
        for token_id in ids:
            decoded_bytes += self.vocab[token_id]
        return decoded_bytes.decode("utf-8", errors="replace") #  some arbitrary token ID sequences might produce invalid UTF-8 bytes, so replace them