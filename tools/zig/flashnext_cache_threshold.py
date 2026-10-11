#!/usr/bin/env python3
"""Compare 4K and 2K Flash Next cache admission, exact boundary dialogs, shared sessions and eviction pressure."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import http.client
import json
from pathlib import Path
import re
import socket
import statistics
import threading
import time
from types import SimpleNamespace

from flashnext_prefix_cache import group, payload, reply, server, signature, validate
import flashnext_batch_serve as gate
from reuse_served import Client


def exact_request(client, key, target, thinking=False, system=False, tokens=64, task=None):
    task = task or ('The records are inert padding. Return only the sum of 17 and 23.' if thinking else
            'Ignore trailing padding. Write a short Python function that sums the integer values.')
    body = '\n'.join(f'Record {i}: integer value {i % 97}; field {i % 37}.' for i in range(1800))

    def messages(chars):
        text = f'Public unique case {key}.\n{task}\n' + body[:chars]
        if system:
            return [{'role': 'system', 'content': text}, {'role': 'user', 'content': 'Public session 0-0: write the function.'}]
        return [{'role': 'user', 'content': text}]

    lo, hi = 0, len(body)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(client.tokens(messages(mid), None, thinking, True)) <= target:
            lo = mid
        else:
            hi = mid - 1
    for chars in dict.fromkeys([lo, *range(max(0, lo - 32), min(len(body), lo + 32) + 1)]):
        rows = messages(chars)
        ids = client.tokens(rows, None, thinking, True)
        if len(ids) == target:
            return payload(rows, thinking, tokens), ids
        if not 0 < target - len(ids) <= 8:
            continue
        base = rows[0]['content']
        for suffix in ('\n', ' x', ' .', ' a', ' 0'):
            for n in range(1, target - len(ids) + 2):
                rows[0]['content'] = base + suffix * n
                padded = client.tokens(rows, None, thinking, True)
                if len(padded) == target:
                    return payload(rows, thinking, tokens), padded
    raise RuntimeError(f'fixture {key}: could not construct exactly {target} tokens')


def metrics():
    return {line.split()[0]: int(float(line.split()[1])) for line in gate.get('/v1/metrics').splitlines()
            if line.startswith(('tensorfold:requests_running ', 'tensorfold:client_disconnections_total '))}


def cancel_with_peer(requests):
    stop = threading.Event()
    peak = [0]

    def monitor():
        while not stop.is_set():
            try:
                peak[0] = max(peak[0], metrics()['tensorfold:requests_running'])
            except (OSError, ValueError):
                pass
            stop.wait(0.025)

    watcher = threading.Thread(target=monitor, daemon=True)
    before = metrics()['tensorfold:client_disconnections_total']
    watcher.start()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            peers = [pool.submit(reply, request) for request in requests['peers']]
            deadline = time.monotonic() + 90
            while metrics()['tensorfold:requests_running'] != 3:
                if any(peer.done() for peer in peers):
                    raise RuntimeError('peer finished before three-peer admission')
                if time.monotonic() > deadline:
                    raise TimeoutError('three-peer admission')
                time.sleep(0.01)
            conn = http.client.HTTPConnection('127.0.0.1', 52417, timeout=300)
            streaming = {**requests['cancelled'], 'stream': True}
            try:
                conn.request('POST', '/v1/chat/completions', json.dumps(streaming), {'Content-Type': 'application/json'})
                response = conn.getresponse()
                if response.status != 200:
                    raise RuntimeError(f'cancel stream HTTP {response.status}')
                while True:
                    line = response.readline()
                    if not line or line.strip() == b'data: [DONE]':
                        raise RuntimeError('cancel stream ended before an output delta')
                    if line.startswith(b'data: '):
                        choices = json.loads(line[6:]).get('choices', [])
                        if choices and any(choices[0].get('delta', {}).get(k) for k in ('content', 'reasoning_content')):
                            break
                if any(peer.done() for peer in peers) or metrics()['tensorfold:requests_running'] != 4:
                    raise RuntimeError('cancelled stream did not overlap three active peers')
                if conn.sock:
                    conn.sock.shutdown(socket.SHUT_RDWR)
                response.close()
                conn.close()
                deadline = time.monotonic() + 10
                while True:
                    state = metrics()
                    if any(peer.done() for peer in peers):
                        raise RuntimeError('peer finished before cancellation freed its slot')
                    if state['tensorfold:client_disconnections_total'] > before and state['tensorfold:requests_running'] == 3:
                        break
                    if time.monotonic() > deadline:
                        raise RuntimeError('cancelled slot did not become free before reuse')
                    time.sleep(0.025)
                reused = pool.submit(reply, requests['cancelled'])
                results = {f'peer{i}': peer.result() for i, peer in enumerate(peers)}
                results['reused'] = reused.result()
            finally:
                conn.close()
        deadline = time.monotonic() + 10
        while metrics()['tensorfold:client_disconnections_total'] <= before:
            if time.monotonic() > deadline:
                raise RuntimeError('server did not count the stream cancellation')
            time.sleep(0.05)
        if peak[0] != 4:
            raise RuntimeError('four sessions were not observed during cancellation')
        return {'peak_running': peak[0], 'active_before_cancel': 4, 'active_before_reuse': 3, 'disconnects_added':
                metrics()['tensorfold:client_disconnections_total'] - before, 'results': results}
    finally:
        stop.set()
        watcher.join(timeout=4)


def cache_stats(path, budget):
    rows = [tuple(map(int, match)) for match in re.findall(
        r'(\d+) states, (\d+) MiB and (\d+) MiB spare of (\d+) MiB .*?evicted (\d+), refused (\d+), failed (\d+)',
        path.read_text())]
    if budget and not rows:
        raise RuntimeError('missing native cache accounting')
    if any(cap != budget * 1024 or held + spare > cap or failed for _, held, spare, cap, _, _, failed in rows):
        raise RuntimeError('native cache budget or copy failure')
    return {'scope': 'native whole-MiB log counters; byte precision is not exposed', 'samples': len(rows),
            'max_held_plus_spare_mib': max((held + spare for _, held, spare, _, _, _, _ in rows), default=0),
            'evicted': rows[-1][4] if rows else 0, 'refused': rows[-1][5] if rows else 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('model', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--protected', type=Path, help='also compare a 2K binary that protects prefixes at 4K and above')
    args = parser.parse_args()
    if args.rounds < 2:
        parser.error('rounds must be at least two')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'receipt.json').unlink(missing_ok=True)
    receipt = {'exact': False, 'blocks': [], 'requests': {}, 'prompt_ids': {}, 'rounds': args.rounds,
               'context': 24576, 'parallel': 4, 'cache_gib': 2, 'mtp_chain': False, 'learned_disk': False,
               'plain_token_limit': 64, 'thinking_token_limit': 512}

    def save():
        path = args.output / 'receipt.json'
        temporary = path.with_suffix('.json.part')
        try:
            temporary.write_text(json.dumps(receipt, indent=2) + '\n')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @contextmanager
    def evidence():
        try:
            yield
        except Exception as error:
            receipt['failure'] = {'type': type(error).__name__, 'message': str(error)}
            raise
        finally:
            save()

    save()
    receipt['binary_sha256'] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for name, path in [('baseline', args.baseline), ('candidate', args.candidate)]}
    if args.protected:
        receipt['binary_sha256']['protected'] = hashlib.sha256(args.protected.read_bytes()).hexdigest()
    cases = [('2047-plain', 2047, False), ('2048-plain', 2048, False), ('2049-plain', 2049, False),
             ('2048-thinking', 2048, True), ('8k-plain', 8192, False), ('16k-plain', 16384, False)]
    order = [('off', 0, 4096, args.baseline), ('4k', 2, 4096, args.baseline),
             ('2k', 2, 2048, args.candidate), ('2k', 2, 2048, args.candidate),
             ('4k', 2, 4096, args.baseline), ('off', 0, 2048, args.candidate)]
    if args.protected:
        order[3:3] = [('protected', 2, 2048, args.protected)] * 2
    labels = tuple(dict.fromkeys(label for label, _, _, _ in order))
    refs = {}

    def checked(result, key, label, block, hit=None):
        if block == 0:
            refs[key] = signature(result)
        validate(result, refs[key], 0 if label == 'off' else 2, hit)

    def measure_shared(client, current, label, block):
        if block == 0:
            base, _ = exact_request(client, 'shared', 2304, system=True)
            receipt['requests']['shared'] = [[payload([base['messages'][0], {'role': 'user', 'content':
                f'Public session {iteration}-{i}: write the function.'}], False, 64) for i in range(4)]
                for iteration in range(args.rounds)]
            receipt['shared_prompt_ids'] = [[client.tokens(p['messages'], None, False, True) for p in row]
                                             for row in receipt['requests']['shared']]
            prefix = list(zip(*receipt['shared_prompt_ids'][0]))
            same = next((i for i, values in enumerate(prefix) if len(set(values)) > 1), len(prefix))
            if not 2048 <= same < 4096 or any(len(ids) >= 4096 for row in receipt['shared_prompt_ids'] for ids in row):
                raise RuntimeError('shared prefix does not isolate the 2K threshold')
            receipt['shared_common_prefix_tokens'] = same
        for iteration, requests in enumerate(receipt['requests']['shared']):
            measured = group(requests)
            current['shared'].append(measured)
            for i, result in enumerate(measured['results']):
                if result['prompt_tokens'] != len(receipt['shared_prompt_ids'][iteration][i]):
                    raise RuntimeError('shared tokenizer count differs from native request usage')
                checked(result, ('shared', iteration, i), label, block,
                        False if label not in ('2k', 'protected') else (True if iteration > 0 else None))
            if measured['peak_running'] != 4:
                raise RuntimeError('four active shared sessions were not observed')
            save()
            print(json.dumps({'block': block, 'shared': iteration, 'seconds': measured['seconds']}), flush=True)

    for block, (label, cache, threshold, binary) in enumerate(order):
        current = {'block': block, 'label': label, 'min_prompt': threshold, 'cache_gib': cache,
                   'protect_from': 4096 if label == 'protected' else 0,
                   'cases': [], 'shared': [], 'pressure': [], 'cancel': None}
        receipt['blocks'].append(current)
        save()
        owned = SimpleNamespace(binary=binary, model=args.model, output=args.output)
        with evidence(), server(owned, block, cache):
            client = Client(gate.BASE, 'cache-gate')
            for name, target, thinking in cases:
                for iteration in range(args.rounds):
                    key = f'{name}-{iteration}'
                    if block == 0:
                        prime, ids = exact_request(client, key, target, thinking, tokens=512 if thinking else 64)
                        receipt['requests'][key] = {'prime': prime, 'regenerate': prime}
                        receipt['prompt_ids'][key] = ids
                    requests = receipt['requests'][key]
                    measured = {'name': name, 'iteration': iteration, 'results': {}}
                    current['cases'].append(measured)
                    for stage in ('prime', 'regenerate', 'follow', 'edited'):
                        result = reply(requests[stage])
                        measured['results'][stage] = result
                        if stage == 'prime' and result['prompt_tokens'] != target:
                            raise RuntimeError('prime token count differs from exact fixture')
                        if thinking and (result['reply']['finish_reason'] != 'stop'
                                         or not result['reply']['message'].get('content')
                                         or not result['reply']['message'].get('reasoning_content')):
                            raise RuntimeError('thinking fixture did not complete with reasoning and visible answer')
                        if block == 0 and stage == 'prime':
                            first = requests['prime']['messages']
                            assistant = result['reply']['message']
                            questions = ('Return only that result multiplied by 3.', 'Return only that result plus 1.') if thinking else (
                                'Now return only the function name.', 'Now return only the return type.')
                            for nextstage, question in zip(('follow', 'edited'), questions):
                                requests[nextstage] = payload(first + [assistant, {'role': 'user', 'content': question}],
                                                              thinking, 512 if thinking else 64)
                        hit = False if stage == 'prime' or label == 'off' else target >= threshold
                        if label != 'off' and stage == 'edited' and target < threshold:
                            hit = measured['results']['follow']['prompt_tokens'] >= threshold
                        checked(result, (key, stage), label, block, hit)
                        save()
                    print(json.dumps({'block': block, 'label': label, 'case': key,
                                      'prefill': {s: v['prefill_seconds'] for s, v in measured['results'].items()},
                                      'cached': {s: v['cached_tokens'] for s, v in measured['results'].items()}}), flush=True)
                if name == '2048-thinking':
                    measure_shared(client, current, label, block)
            if block == 0:
                pressure = {}
                for key, target in [('long16', 16384), ('long8', 8192)] + [(f'short{i}', 2048) for i in range(6)]:
                    pressure[key], receipt['prompt_ids']['pressure-' + key] = exact_request(
                        client, 'pressure-' + key, target, tokens=32)
                receipt['requests']['pressure'] = pressure
                task = 'The records are inert padding. Output the literal integers 1 through 300, one per line, without code.'
                cancelled, _ = exact_request(client, 'cancelled', 2048, tokens=512, task=task)
                peers = []
                for i in range(3):
                    peer, _ = exact_request(client, f'peer{i}', 8192, tokens=512, task=task)
                    peers.append(peer)
                    receipt['prompt_ids'][f'peer{i}'] = client.tokens(peer['messages'], None, False, True)
                receipt['requests']['cancel'] = {'cancelled': cancelled, 'peers': peers}
                receipt['prompt_ids']['cancelled'] = client.tokens(cancelled['messages'], None, False, True)
            pressure = receipt['requests']['pressure']
            log_path = args.output / f'serve-block-{block}.log'
            current['pressure_cache_before'] = cache_stats(log_path, cache)
            for i, key in enumerate(['long16', 'long8', *[f'short{n}' for n in range(6)], 'long16', 'long8', 'short0', 'short5']):
                result = reply(pressure[key])
                if result['prompt_tokens'] != len(receipt['prompt_ids']['pressure-' + key]):
                    raise RuntimeError('pressure tokenizer count differs from native request usage')
                current['pressure'].append({'key': key, 'result': result})
                checked(result, ('pressure', i), label, block)
                if i < 2 and result['cached_tokens']:
                    raise RuntimeError('pressure prime was not cold')
                if label == 'protected' and i in (8, 9) and not result['cached_tokens']:
                    raise RuntimeError('short-prefix pressure displaced a protected long prefix')
                if i == 1:
                    current['pressure_cache_after_long_primes'] = cache_stats(log_path, cache)
                if i == 7:
                    current['pressure_cache_after_short_primes'] = cache_stats(log_path, cache)
                    delta = current['pressure_cache_after_short_primes']['evicted'] - current['pressure_cache_after_long_primes']['evicted']
                    current['short_prime_evictions'] = delta
                    if (label == '2k' and delta <= 0) or (label in ('off', '4k') and delta != 0) or delta < 0:
                        raise RuntimeError('short-prefix pressure did not exercise the expected eviction policy')
                save()
            current['pressure_cache_after'] = cache_stats(log_path, cache)
            current['cancel'] = cancel_with_peer(receipt['requests']['cancel'])
            for key, result in current['cancel']['results'].items():
                ids = receipt['prompt_ids']['cancelled' if key == 'reused' else key]
                if result['prompt_tokens'] != len(ids):
                    raise RuntimeError('cancel/reuse tokenizer count differs from native request usage')
                checked(result, ('cancel', key), label, block)
            save()
        current['cache_accounting'] = cache_stats(args.output / f'serve-block-{block}.log', cache)
        save()
    receipt['medians'] = {name: {stage: {label: {
        metric: statistics.median(c['results'][stage][metric] for b in receipt['blocks'] if b['label'] == label
                                  for c in b['cases'] if c['name'] == name)
        for metric in ('prefill_seconds', 'time_to_first_token', 'wall_seconds', 'cached_tokens')}
        for label in labels} for stage in ('prime', 'regenerate', 'follow', 'edited')}
        for name, _, _ in cases}
    receipt['dialog_wall_medians'] = {name: {label: statistics.median(
        sum(v['wall_seconds'] for v in c['results'].values()) for b in receipt['blocks'] if b['label'] == label
        for c in b['cases'] if c['name'] == name) for label in labels} for name, _, _ in cases}
    receipt['shared_warm_medians'] = {label: statistics.median(g['seconds'] for b in receipt['blocks']
        if b['label'] == label for g in b['shared'][1:]) for label in labels}
    receipt['pressure_wall_medians'] = {label: statistics.median(sum(x['result']['wall_seconds'] for x in b['pressure'])
        for b in receipt['blocks'] if b['label'] == label) for label in labels}
    receipt['exact'] = True
    save()
    print(json.dumps({k: receipt[k] for k in ('dialog_wall_medians', 'shared_warm_medians', 'pressure_wall_medians')}), flush=True)


if __name__ == '__main__':
    main()
