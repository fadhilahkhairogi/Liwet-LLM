from typing import Optional, Tuple

from tokenizers import Tokenizer

ROLE_LABEL = {"system": "System", "user": "User", "assistant": "Assistant"}


def render_messages(messages: list, assistant_prefix: bool = False) -> str:
    parts = [
        f"{ROLE_LABEL[message['role']]}: {message['content']}" for message in messages
    ]
    if assistant_prefix:
        parts.append("Assistant:")
    return "\n\n".join(parts)


def render_prompt(messages: list) -> str:
    return render_messages(messages, assistant_prefix=True)


def token_ids(tokenizer: Tokenizer) -> Tuple[int, int, int]:
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    pad = tokenizer.token_to_id("<pad>")
    if None in (bos, eos, pad):
        raise ValueError("tokenizer harus memiliki <bos>, <eos>, dan <pad>")
    return bos, eos, pad


def encode_sft(
    tokenizer: Tokenizer, messages: list, max_len: int
) -> Optional[Tuple[list, list]]:
    bos_id, eos_id, _ = token_ids(tokenizer)
    parts = []
    assistant_spans = []
    cursor = 0

    for index, message in enumerate(messages):
        prefix = f"{ROLE_LABEL[message['role']]}: "
        parts.append(prefix)
        cursor += len(prefix)

        content = message["content"]
        start = cursor
        parts.append(content)
        cursor += len(content)
        end = cursor

        if message["role"] == "assistant":
            assistant_spans.append((start, end))

        if index < len(messages) - 1:
            parts.append("\n\n")
            cursor += 2

    text = "".join(parts)
    encoding = tokenizer.encode(text, add_special_tokens=False)
    input_ids = [bos_id] + encoding.ids + [eos_id]
    labels = [-100] * (len(encoding.ids) + 1) + [eos_id]

    for token_index, (start, end) in enumerate(encoding.offsets):
        for span_start, span_end in assistant_spans:
            if start < span_end and end > span_start:
                labels[1 + token_index] = encoding.ids[token_index]
                break

    if not any(label != -100 for label in labels):
        return None

    if len(input_ids) > max_len:
        input_ids = input_ids[: max_len - 1] + [eos_id]
        labels = labels[: max_len - 1] + [eos_id]

    return input_ids, labels


def encode_preference(
    tokenizer: Tokenizer, prompt: str, response: str, max_len: int
) -> Optional[Tuple[list, list]]:
    bos_id, eos_id, _ = token_ids(tokenizer)
    separator = "" if prompt.endswith(" ") else " "
    full_text = prompt + separator + response
    response_start = len(prompt)

    encoding = tokenizer.encode(full_text, add_special_tokens=False)
    input_ids = [bos_id] + encoding.ids + [eos_id]
    labels = [-100] * (len(encoding.ids) + 1) + [eos_id]

    for token_index, (start, end) in enumerate(encoding.offsets):
        if start >= response_start:
            labels[1 + token_index] = encoding.ids[token_index]

    if not any(label != -100 for label in labels):
        return None

    if len(input_ids) > max_len:
        input_ids = input_ids[: max_len - 1] + [eos_id]
        labels = labels[: max_len - 1] + [eos_id]

    return input_ids, labels
