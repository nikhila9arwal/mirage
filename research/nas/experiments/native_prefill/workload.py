"""Tokenize real prompts locally. NAS receives tokens, never an HF KV prefix."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'qwen_decoder_layer'))
from prepare_prompt import _checkpoint, TEMPLATE_KWARGS
from reference import DEFAULT_CHECKPOINT

PAGE = 64


def signature(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def questions():
    """64 distinct short requests; selection below makes complete P-row chunks."""
    facts = [
        'What is the capital of France?', 'What is the capital of Japan?',
        'What is the capital of Italy?', 'What is the capital of Canada?',
        'Which planet is called the red planet?', 'What gas do plants absorb?',
        'How many days are in a week?', 'How many sides does a triangle have?',
        'What color do blue and yellow paint make?', 'Which ocean is the largest?',
        'What is the chemical symbol for gold?', 'What is the opposite of north?',
        'What is the freezing point of water in Celsius?', 'Name the largest land animal.',
        'What is the smallest prime number?', 'How many months are in a year?',
    ]
    arithmetic = [f'What is {a} plus {b}?' for a, b in
                  [(2,3),(4,5),(6,7),(8,9),(3,8),(9,4),(12,5),(15,6)]]
    arithmetic += [f'What is {a} times {b}?' for a, b in
                   [(2,4),(3,5),(4,6),(5,7),(6,8),(7,9),(8,3),(9,2)]]
    language = [f'Translate the English word "{word}" into Spanish.'
                for word in ['cat','dog','water','bread','red','blue','sun','moon']]
    definitions = [f'Define {word} in one short sentence.' for word in
                   ['a compiler','a database','gravity','photosynthesis',
                    'evaporation','an algorithm','a triangle','a volcano']]
    lists = [f'Name two {category}.' for category in
             ['fruits','vegetables','mammals','birds','musical instruments',
              'European countries','programming languages','primary colors']]
    short = [
        'Write a friendly greeting.', 'Write a brief thank-you message.',
        'Describe snow in three words.', 'Describe the ocean in three words.',
        'Complete the phrase: better late than', 'Complete the phrase: practice makes',
        'What comes after Monday?', 'What comes before December?',
    ]
    return facts + arithmetic + language + definitions + lists + short


def validate(data):
    if data.get('format') != 'qwen_native_prefill_v1':
        raise ValueError('Unknown native-prefill workload')
    p, vocab = data['chunk_size'], data['vocab_size']
    if p not in (1, 2, 4) or not 0 <= data['eos_token_id'] < vocab:
        raise ValueError('Invalid chunk capacity or EOS')
    if not data['prompts']:
        raise ValueError('At least one prompt is required')
    for index, row in enumerate(data['prompts']):
        ids = row['token_ids']
        if row['request'] != index or len(ids) != row['length']:
            raise ValueError('Request index or length mismatch')
        if not 1 <= len(ids) <= PAGE or len(ids) % p:
            raise ValueError('Prompts must fit the cache and contain full chunks')
        if any(type(token) is not int or not 0 <= token < vocab for token in ids):
            raise ValueError('Invalid prompt token')
    expected = signature({key: value for key, value in data.items() if key != 'sha256'})
    if data.get('sha256') != expected:
        raise ValueError('Workload fingerprint mismatch')
    return data


def prepare(checkpoint, chunk_size, supplied=None):
    from transformers import AutoTokenizer
    manifest, snapshot, config = _checkpoint(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    texts = questions() if supplied is None else supplied
    if not isinstance(texts, list) or not all(isinstance(x, str) for x in texts):
        raise ValueError('Prompts must be a JSON list of strings')
    # Choose an explicit natural-language instruction whose token count fits.
    # Supplied prompts are never modified; a partial chunk is rejected instead.
    suffixes = ['', ' Answer briefly.', ' Be concise.', ' Give a short answer.',
                ' Reply in one sentence.', ' Please answer.', ' Use a few words.',
                ' Please be brief.', ' Give only the answer.',
                ' Keep the answer short.', ' Answer in English.',
                ' Please give a concise answer.', ' Respond briefly in English.']
    rows = []
    for request, question in enumerate(texts):
        for suffix in (suffixes if supplied is None else ['']):
            text = question + suffix
            rendered = tokenizer.apply_chat_template(
                [dict(role='user', content=text)], tokenize=False, **TEMPLATE_KWARGS)
            ids = tokenizer(rendered, add_special_tokens=False, truncation=False,
                            padding=False)['input_ids']
            if 1 <= len(ids) <= (40 if supplied is None else PAGE) and len(ids) % chunk_size == 0:
                rows.append(dict(request=request, text=text, rendered=rendered,
                                 token_ids=ids, length=len(ids)))
                break
        else:
            raise ValueError(f'No full-chunk prompt fits request {request}: {question}')
    data = dict(format='qwen_native_prefill_v1', model=manifest['model'], revision=manifest['revision'],
                checkpoint=str(Path(checkpoint).resolve()), chunk_size=chunk_size,
                vocab_size=config['vocab_size'], eos_token_id=config['eos_token_id'],
                capacity=PAGE, template_kwargs=TEMPLATE_KWARGS, prompts=rows,
                prefill='NAS starts with an empty private KV cache; no HF prefix is supplied')
    data['sha256'] = signature(data)
    return validate(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--chunk-size', type=int, choices=(1,2,4), default=4)
    parser.add_argument('--prompts', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    supplied = json.loads(args.prompts.read_text()) if args.prompts else None
    data = prepare(args.checkpoint, args.chunk_size, supplied)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps(dict(requests=len(data['prompts']), sha256=data['sha256'],
                          lengths=[row['length'] for row in data['prompts']]), indent=2))


if __name__ == '__main__':
    main()
