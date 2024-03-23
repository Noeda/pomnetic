#!/usr/bin/env python

import io
import time
import json
import sys

# Sync with Haskell code
# Used to detect lines that are interpreted on Haskell side (as opposed random
# stuff possibly output from transformers or other modules)
MAGIC = 'dQ25CNJGb94QejK0'

def main():
    try:
        import transformers
    except ImportError:
        sys.stderr.buffer.write((f'\n{MAGIC} TRANSFORMERS_IMPORT_ERROR\n').encode('utf-8'))
        sys.stderr.buffer.flush()
        sys.exit(1)

    # map from
    # (model_name, local_files_only, trust_remote_code) -> (tokenizer, last_used)
    loaded_tokenizers = {}
    loaded_tokenizers_gc_cycle = 100

    sys.stdout.flush()
    sys.stdout.buffer.write((f'\n{MAGIC} READY\n').encode('utf-8'))
    sys.stdout.buffer.flush()

    last_gc_check = None

    while True:
        if loaded_tokenizers_gc_cycle <= 0 or last_gc_check is None or (time.monotonic() - last_gc_check) > 60:
            loaded_tokenizers_gc_cycle = 100
            last_gc_check = time.monotonic()

            # Remove tokenizers not used in last 1 minute
            now = time.monotonic()
            loaded_tokenizers = {k: v for k, v in loaded_tokenizers.items() if now - v[1] < 60}

        loaded_tokenizers_gc_cycle -= 1

        msg = sys.stdin.readline()
        if msg == '':
            break

        msg = msg.strip()
        val = json.loads(msg)
        if val['type'] == 'metadata':
            model = val['model']

            key = (model, False, False)
            now = time.monotonic()
            if key not in loaded_tokenizers:
                loaded_tokenizers[key] = (transformers.AutoTokenizer.from_pretrained(model), now)

            bos_token = loaded_tokenizers[key][0].bos_token_id
            eos_token = loaded_tokenizers[key][0].eos_token_id
            sep_token = loaded_tokenizers[key][0].sep_token_id

            answer = { 'py_bos_token': bos_token, 'py_eos_token': eos_token, 'py_sep_token': sep_token}

            sys.stdout.buffer.write((f'\n{MAGIC} {json.dumps(answer)}\n').encode('utf-8'))
            sys.stdout.buffer.flush()

        elif val['type'] == 'tokenize':
            tokenize_text = val['text']
            model = val['model']
            add_special_tokens = val['add_special_tokens']
            local_files_only = val['local_files_only']
            trust_remote_code = val['trust_remote_code']

            now = time.monotonic()

            key = (model, local_files_only, trust_remote_code)
            if key not in loaded_tokenizers:
                loaded_tokenizers[key] = (transformers.AutoTokenizer.from_pretrained(model, local_files_only=local_files_only), now)
            tokenizer, _ = loaded_tokenizers[key]
            loaded_tokenizers[key] = (tokenizer, now)

            # Some kind of integer I see in HF models to indicate there is no
            # max length. Some limit at precision? Not sure. We'll use it
            # anyway.
            max_len = 1000000000000000019884624838656

            tokenized = tokenizer(tokenize_text, add_special_tokens=add_special_tokens, max_length=max_len, truncation=False)
            tokenized = tokenized['input_ids']

            answer = { 'tokens': tokenized }
            sys.stdout.buffer.write((f'\n{MAGIC} {json.dumps(answer)}\n').encode('utf-8'))
            sys.stdout.buffer.flush()

if __name__ == '__main__':
    main()
