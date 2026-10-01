"""The shared training and inference answer-slot prompt."""
import json
import string

ANSWER_CUE = 'The correct answer is ('


def text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def option_labels(tokenizer):
    candidates = list(string.ascii_uppercase) + [a+b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    labels = [label for label in candidates
              if len(tokenizer.encode(label, add_special_tokens=False)) == 1][:255]
    if len(labels) != 255:
        raise ValueError('The tokenizer must provide the original 255 single-token option labels.')
    return labels


def body(row, labels):
    choices = '\n'.join(f'({labels[i]}) {o["key"]}: {o["desc"]}' for i, o in enumerate(row['options']))
    return f'State:\n{text(row["state"])}\nQuestion: {text(row["question"])}\nOptions:\n{choices}'.rstrip()


def prompt(tokenizer, row, labels):
    prefix = tokenizer.apply_chat_template([{'role': 'user', 'content': body(row, labels)}],
                                          tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False)
    return prefix + ANSWER_CUE
