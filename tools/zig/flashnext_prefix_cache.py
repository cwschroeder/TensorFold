#!/usr/bin/env python3
"""Measure exact Flash Next chat reuse with alternating cache-off/on blocks and a RAM-only restart control."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
import urllib.request

import flashnext_batch_serve as gate
from reuse_served import Client


def reply(payload, barrier=None):
    if barrier:
        barrier.wait()
    begin = time.monotonic()
    req = urllib.request.Request(gate.BASE + '/v1/chat/completions', json.dumps(payload).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as response:
        result = json.load(response)
    usage, runtime = result['usage'], result['tensorfold']
    prompt, cached, tokens = (usage['prompt_tokens'], usage['prompt_tokens_details']['cached_tokens'],
                              usage['completion_tokens'])
    sha = runtime.get('token_sha')
    if any(type(x) is not int for x in (prompt, cached, tokens)) or not 0 <= cached < prompt or tokens < 1:
        raise RuntimeError('invalid native usage')
    if not isinstance(sha, str) or len(sha) != 12 or any(c not in '0123456789abcdef' for c in sha):
        raise RuntimeError('missing native token fingerprint')
    timings = {key: runtime.get(key) for key in ('prefill_seconds', 'time_to_first_token', 'seconds')}
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in timings.values()):
        raise RuntimeError('missing native timing')
    return {'reply': result['choices'][0], 'prompt_tokens': prompt, 'cached_tokens': cached,
            'tokens': tokens, 'token_sha': sha, **timings, 'wall_seconds': time.monotonic() - begin}


def signature(result):
    return {key: result[key] for key in ('reply', 'prompt_tokens', 'tokens', 'token_sha')}


def payload(messages, thinking, tokens):
    return {'model': 'cache-gate', 'messages': messages, 'max_tokens': tokens, 'temperature': 0,
            'stream': False, 'chat_template_kwargs': {'enable_thinking': thinking}}


@contextmanager
def server(args, block, cache):
    path = args.output / f'serve-block-{block}.log'
    with path.open('w') as log:
        process = subprocess.Popen([str(args.binary), 'serve', str(args.model), '--name', 'cache-gate',
                                    '--host', '127.0.0.1', '--port', '52417', '--context', '24576',
                                    '--parallel', '4', '--temperature', '0', '--prompt-cache-gib', str(cache),
                                    '--no-update-check'], stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'FZ_BATCH_MTP_CHAIN': '0'})
        (args.output / 'server.pid').write_text(str(process.pid) + '\n')
        try:
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'native server exited {process.returncode}: {path.name}')
                try:
                    health = json.loads(gate.get('/health'))
                    if health.get('status') == 'ok' and health.get('warming') is False:
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(0.2)
            else:
                raise TimeoutError('native startup')
            yield
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
            (args.output / 'server.pid').unlink(missing_ok=True)


def group(requests):
    stop = threading.Event()
    peak = [0]

    def monitor():
        while not stop.is_set():
            try:
                for line in gate.get('/v1/metrics').splitlines():
                    if line.startswith('tensorfold:requests_running '):
                        peak[0] = max(peak[0], int(float(line.split()[1])))
            except (OSError, ValueError):
                pass
            stop.wait(0.025)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    start = time.monotonic()
    try:
        barrier = threading.Barrier(len(requests))
        with ThreadPoolExecutor(max_workers=len(requests)) as pool:
            futures = [pool.submit(reply, request, barrier) for request in requests]
            results = [future.result() for future in futures]
        return {'seconds': time.monotonic() - start, 'peak_running': peak[0], 'results': results}
    finally:
        stop.set()
        watcher.join(timeout=4)


def validate(result, reference, cache, hit=None):
    if signature(result) != reference:
        raise RuntimeError('cache-off/on reply, token fingerprint or usage differs')
    if cache == 0 and result['cached_tokens'] != 0:
        raise RuntimeError('cache-off restored tokens')
    if hit is not None and (result['cached_tokens'] > 0) != hit:
        raise RuntimeError(f"unexpected prefix reuse: cached={result['cached_tokens']}, expected hit={hit}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    parser.add_argument('model', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--tokens', type=int, default=64)
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    if args.tokens < 1 or args.rounds < 2:
        parser.error('tokens must be positive and rounds at least two')
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {'exact': False, 'blocks': [], 'requests': {}, 'reply_token_limit': args.tokens,
               'rounds': args.rounds, 'context': 24576, 'parallel': 4, 'cache_gib': 2,
               'mtp_chain': False, 'learned_disk': False}

    def save():
        (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')

    save()
    receipt['binary_sha256'] = hashlib.sha256(args.binary.read_bytes()).hexdigest()
    cases = [('2k-plain', 2048, False), ('8k-plain', 8192, False), ('8k-thinking', 8192, True),
             ('16k-plain', 16384, False), ('16k-thinking', 16384, True)]
    references = {}
    shared_requests = [[payload([{'role': 'system', 'content': '\n'.join(
        f'Policy row {i}: read field {i % 37} and preserve integer values.' for i in range(500))},
        {'role': 'user', 'content': f'Public fresh session {iteration}-{i}: write a short Python function to sum record field {i}.'}],
        False, args.tokens) for i in range(4)] for iteration in range(args.rounds)]
    receipt['requests']['shared'] = shared_requests
    shared_reference = {}
    for block, cache in enumerate((0, 2, 2, 0)):
        current = {'block': block, 'cache_gib': cache, 'cases': [], 'shared': []}
        receipt['blocks'].append(current)
        save()
        with server(args, block, cache):
            for name, target, thinking in cases:
                for iteration in range(args.rounds):
                    key = f'{name}-{iteration}'
                    if block == 0:
                        client = Client(gate.BASE, 'cache-gate')
                        body = '\n'.join(f'Record {i}: integer value {i % 97}; field {i % 37}.'
                                         for i in range(1800))

                        def messages(chars):
                            return [{'role': 'user', 'content': f'Public case {key}.\n' + body[:chars]
                                     + '\nWrite a short Python function that sums the integer values.'}]

                        lo, hi = 0, len(body)
                        while lo < hi:
                            mid = (lo + hi + 1) // 2
                            if len(client.tokens(messages(mid), None, thinking, True)) <= target:
                                lo = mid
                            else:
                                hi = mid - 1
                        prime = payload(messages(lo), thinking, args.tokens)
                        ids = client.tokens(prime['messages'], None, thinking, True)
                        if not target - 4 <= len(ids) <= target:
                            raise RuntimeError('public fixture misses target token length')
                        receipt['requests'][key] = {'prime': prime, 'regenerate': prime}
                        receipt.setdefault('prompt_ids', {})[key] = ids
                    messages_ = receipt['requests'][key]['prime']['messages']
                    requests = receipt['requests'][key]
                    measured = {'name': name, 'iteration': iteration, 'results': {}}
                    current['cases'].append(measured)
                    for stage in ('prime', 'regenerate', 'follow', 'edited'):
                        result = reply(requests[stage])
                        measured['results'][stage] = result
                        if stage == 'prime' and result['prompt_tokens'] != len(receipt['prompt_ids'][key]):
                            raise RuntimeError('tokenization differs from native request usage')
                        refkey = (key, stage)
                        if block == 0:
                            references[refkey] = signature(result)
                            if stage == 'prime':
                                assistant = result['reply']['message']
                                for nextstage, question in (('follow', 'Now return only the function name.'),
                                                            ('edited', 'Now return only the return type.')):
                                    requests[nextstage] = payload(messages_ + [assistant,
                                        {'role': 'user', 'content': question}], thinking, args.tokens)
                        validate(result, references[refkey], cache,
                                 hit=False if stage == 'prime' else (cache > 0 and name != '2k-plain'))
                        save()
                    print(json.dumps({'block': block, 'cache': cache, 'case': key,
                                      'results': measured['results']}), flush=True)
            for iteration in range(args.rounds):
                measured = group(shared_requests[iteration])
                current['shared'].append(measured)
                if block == 0:
                    shared_reference[iteration] = [signature(result) for result in measured['results']]
                for result, reference in zip(measured['results'], shared_reference[iteration]):
                    validate(result, reference, cache, hit=True if cache > 0 and iteration > 0 else None)
                if measured['peak_running'] != 4:
                    raise RuntimeError('four simultaneous shared sessions were not observed')
                save()
                print(json.dumps({'block': block, 'shared_round': iteration, **measured}), flush=True)
    summary = {}
    for name, _, _ in cases:
        summary[name] = {}
        for stage in ('prime', 'regenerate', 'follow', 'edited'):
            summary[name][stage] = {str(cache): {
                metric: statistics.median(case['results'][stage][metric] for block in receipt['blocks']
                    if block['cache_gib'] == cache for case in block['cases'] if case['name'] == name)
                for metric in ('prefill_seconds', 'time_to_first_token', 'wall_seconds', 'cached_tokens')}
                for cache in (0, 2)}
    receipt['shared_warm_medians'] = {str(cache): statistics.median(
        batch['seconds'] for block in receipt['blocks'] if block['cache_gib'] == cache
        for batch in block['shared'][1:]) for cache in (0, 2)}
    receipt['medians'] = summary
    receipt['exact'] = True
    save()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
