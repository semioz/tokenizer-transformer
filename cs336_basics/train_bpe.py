from collections import Counter
from multiprocessing import Pool
import os
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(file, desired_num_chunks, split_special_token):
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)

        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break

            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def pretokenize_text(text, special_tokens):
    special_pattern = "|".join(re.escape(tok) for tok in special_tokens)
    parts = re.split(special_pattern, text) if special_tokens else [text]

    word_counts = Counter()

    for part in parts:
        for match in re.finditer(PAT, part):
            token_bytes = match.group().encode("utf-8")
            word = tuple(bytes([b]) for b in token_bytes)
            word_counts[word] += 1

    return word_counts


def pretokenize_chunk(args):
    input_path, start, end, special_tokens = args

    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")

    return pretokenize_text(text, special_tokens)


def pretokenize_file(input_path, special_tokens, num_processes=None):
    if num_processes is None:
        file_size = os.path.getsize(input_path)
        num_processes = min(os.cpu_count() or 1, 8) if file_size >= 10_000_000 else 1

    if num_processes <= 1 or not special_tokens:
        with open(input_path, "r", encoding="utf-8") as f:
            return pretokenize_text(f.read(), special_tokens)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))

    jobs = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    if len(jobs) <= 1:
        return pretokenize_chunk(jobs[0]) if jobs else Counter()

    with Pool(processes=min(num_processes, len(jobs))) as pool:
        chunk_counts = pool.map(pretokenize_chunk, jobs)

    word_counts = Counter()
    for counts in chunk_counts:
        word_counts.update(counts)

    return word_counts

def merge_word(word, pair):
    is_changed = False
    merged_word = []
    i = 0

    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            merged_word.append(word[i] + word[i + 1])
            is_changed = True
            i += 2
        else:
            merged_word.append(word[i])
            i += 1

    if is_changed:
        return tuple(merged_word)

    return word


def get_word_pair_counts(word):
    pair_counts = Counter()

    for i in range(len(word) - 1):
        pair_counts[(word[i], word[i + 1])] += 1

    return pair_counts

def get_pair_counts(word_counts):
    pair_counts = Counter()

    for word, count in word_counts.items():
        for pair, pair_count in get_word_pair_counts(word).items():
            pair_counts[pair] += pair_count * count

    return pair_counts


def build_pair_indexes(word_counts):
    pair_counts = Counter()
    pair_to_words = {}

    for word, word_count in word_counts.items():
        for pair, pair_count in get_word_pair_counts(word).items():
            pair_counts[pair] += pair_count * word_count
            pair_to_words.setdefault(pair, set()).add(word)

    return pair_counts, pair_to_words

def build_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1
    return vocab

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        vocab = build_vocab(special_tokens)
        word_counts = pretokenize_file(input_path, special_tokens)

        merges = []
        num_merges = vocab_size - len(vocab)
        pair_counts, pair_to_words = build_pair_indexes(word_counts)
        
        for _ in range(num_merges):
            if not pair_counts:
                break
                
            best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
    
            a, b  = best_pair
            merged = a + b
            vocab[len(vocab)] = merged
            merges.append((a, b))

            affected_words = list(pair_to_words.get(best_pair, set()))

            for word in affected_words:
                count = word_counts.get(word)
                if count is None:
                    continue

                old_pair_counts = get_word_pair_counts(word)
                if best_pair not in old_pair_counts:
                    continue

                del word_counts[word]

                for pair, pair_count in old_pair_counts.items():
                    pair_counts[pair] -= pair_count * count
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]

                    if pair in pair_to_words:
                        pair_to_words[pair].discard(word)
                        if not pair_to_words[pair]:
                            del pair_to_words[pair]

                new_word = merge_word(word, best_pair)
                word_counts[new_word] += count

                for pair, pair_count in get_word_pair_counts(new_word).items():
                    pair_counts[pair] += pair_count * count
                    pair_to_words.setdefault(pair, set()).add(new_word)

        return vocab, merges
