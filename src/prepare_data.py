import common
import scrape
from log import (
    important,
    log,
    pretty_format,
    ProgressStatus,
    set_status_state,
    set_substeps,
    warning,
    LogTypes,
)

from collections import Counter
import os
from enum import Enum

vowels = [c for c in "aeiouyåäö".upper()] + [c for c in "aeiouyåäö"]
vocab_size = 1_500
word_amount = 500
min_word_length = 4
max_syllable_length = 2


def split_to_words(post):
    post = [post]
    separators = [" ", ":", ";", "=", ",", ".", "!", "?", "-", '"']
    for s in separators:
        post = [w for p in post for w in p.split(s) if len(w) >= min_word_length]
    return post


def words_to_lookup(words):
    d = {}
    for w in words:
        c = w[0:min_word_length]
        if c not in d:
            d[c] = []
        d[c].append(w)
    return d


def split_to_characters(common_words, post):
    all_parts = []
    current_part = []
    word_remaining = 0
    for i, c in enumerate(post):
        if word_remaining == -1:
            all_parts.append("".join(current_part))
            current_part = []
            word = next(
                (
                    w
                    for w in common_words.get(post[i : i + min_word_length], [])
                    if post.startswith(w, i)
                ),
                None,
            )
            if word is not None:
                word_remaining = len(word)

        if word_remaining > 0:
            word_remaining -= 1

        if word_remaining == -1 and (
            c not in vowels or len(current_part) > max_syllable_length
        ):
            if len(current_part):
                all_parts.append("".join(current_part))
                current_part = []

        current_part.append(c)

        if word_remaining == 0:
            word_remaining = -1
            all_parts.append("".join(current_part))
            current_part = []

    if len(current_part):
        all_parts.append("".join(current_part))
    return all_parts


class SpecialCharacters(Enum):
    EMPTY = "empty"
    NEW_TOPIC = "new_topic"
    NEW_POST = "new_post"
    VOCAB_PADDING = "vocab_padding"


if __name__ == "__main__":
    important("Preparing training data")

    dataset_chunk_dir = os.path.join(common.dataset_dir, "chunks")
    common.ensure_dir(dataset_chunk_dir)

    dataset_filenames = [
        filename for filename in common.list_all_files(scrape.dataset_post_dir)
    ]
    dataset_filenames.sort()

    def load_topic(filename):
        log(f"Loading {filename}", repeating_status=True)
        return common.load_json_file(filename)

    important("Loading topics")
    set_status_state(ProgressStatus(len(dataset_filenames)))
    data = [load_topic(filename) for filename in dataset_filenames]

    set_status_state(ProgressStatus(len(data)))
    all_words = Counter()
    for topic in data:
        log("Counting unique words", repeating_status=True)
        set_substeps(len(topic))
        for post in topic:
            log("Counting unique words", repeating_status=True, substep=True)
            for word in list(set(split_to_words(post))):
                all_words[word] += 1

    most_common_words = [word for (word, count) in all_words.most_common(word_amount)]
    word_lookup = words_to_lookup(most_common_words)
    important(f"Most common words: {', '.join(most_common_words[:5])}")

    set_status_state(ProgressStatus(len(data)))
    all_characters = Counter()
    for topic in data:
        log(
            f"Counting unique characters ({len(all_characters)})", repeating_status=True
        )
        set_substeps(len(topic))
        for post in topic:
            log(
                f"Counting unique characters ({len(all_characters)})",
                repeating_status=True,
                substep=True,
            )
            for char in list(set(split_to_characters(word_lookup, post))):
                if char not in most_common_words:
                    all_characters[char] += 1
                    if len(char) > 3:
                        log(f'"{char}"')

    important(f"Unique character count: {len(all_characters)}")

    chars_by_usage = []
    most_common = all_characters.most_common()

    log("Listing characters")
    unused_vocab = (
        vocab_size - len(most_common_words) - len(most_common) - len(SpecialCharacters)
    )
    most_common_words.reverse()
    chars_by_usage = (
        [c.value for c in SpecialCharacters]
        + [word for (word, _) in most_common]
        + [SpecialCharacters.VOCAB_PADDING.value for i in range(unused_vocab)]
        + [word for word in most_common_words]
    )
    log(f"Unused slots in vocab remaining: {unused_vocab}")

    log("Creating character to index mappings")
    char_to_index = {c: i for (i, c) in enumerate(chars_by_usage)}
    index_to_char = {i: c for (c, i) in char_to_index.items()}

    def post_to_index(post):
        post = split_to_characters(word_lookup, post)
        ids = [char_to_index[c] for c in [SpecialCharacters.NEW_POST.value] + post]
        return ids

    log("Writing characters.json")
    common.write_json_file(
        os.path.join(common.cache_dir, "characters.json"),
        {
            "characters": chars_by_usage,
            "words": most_common_words,
            "char_to_index": char_to_index,
            "index_to_char": index_to_char,
        },
    )

    important("Converting posts to training data")

    set_status_state(ProgressStatus(len(data)))
    for i, topic in enumerate(data):
        post_indexes = [n for post in topic for n in post_to_index(post)]
        filename = f"{i}_posts.json"
        log(f"Writing {filename}", repeating_status=True)
        common.ensure_dir(os.path.join(common.cache_dir, "posts"))
        common.write_json_file(
            os.path.join(common.cache_dir, "posts", filename), post_indexes
        )

    important("Done")
