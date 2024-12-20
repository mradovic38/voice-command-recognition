import json

def extract_all_chars(words):
    all_text = "".join(words)
    vocab = list(set(all_text))
    return vocab


def save_dict_as_json(path, dict_to_save):
    with open(path, 'w') as vocab_file:
        json.dump(dict_to_save, vocab_file)