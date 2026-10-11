import collections,gzip,json,math,pathlib,re,statistics,sys
path=pathlib.Path(sys.argv[1]); data=path.read_bytes(); r=json.loads(gzip.decompress(data) if path.suffix=='.gz' else data)
pins=json.loads((path.parent/'build-pins.json').read_text())
assert r['binary_sha256']=={key:pins['binaries'][binary]['sha256'] for key,binary in [('baseline','native-4k'),('candidate','native-2k'),('protected','native-protected')]}
assert r['exact'] is True and len(r['blocks'])==8
assert [b['label'] for b in r['blocks']]==['off','4k','2k','protected','protected','2k','4k','off']
assert [b['cache_gib'] for b in r['blocks']]==[0,2,2,2,2,2,2,0]
assert [b['min_prompt'] for b in r['blocks']]==[4096,4096,2048,2048,2048,2048,4096,2048]
assert r['context']==24576 and r['parallel']==4 and r['rounds']==3 and r['mtp_chain'] is False and r['learned_disk'] is False
targets={'2047-plain':2047,'2048-plain':2048,'2049-plain':2049,'2048-thinking':2048,'8k-plain':8192,'16k-plain':16384}
refs={};counts=collections.Counter();blocks=[]
sig=lambda v:{k:v[k] for k in ('reply','prompt_tokens','tokens','token_sha')}
for expected_block,b in enumerate(r['blocks']):
    assert b['block']==expected_block and len(b['cases'])==18
    assert collections.Counter((c['name'],c['iteration']) for c in b['cases'])==collections.Counter((name,i) for name in targets for i in range(3))
    accounting=b['cache_accounting']
    log=path.parent/f'serve-block-{expected_block}.log.gz'
    raw=gzip.decompress(log.read_bytes()).decode()
    samples=[tuple(map(int,m)) for m in re.findall(r'(\d+) states, (\d+) MiB and (\d+) MiB spare of (\d+) MiB .*?evicted (\d+), refused (\d+), failed (\d+)',raw)]
    assert bool(samples)==bool(b['cache_gib'])
    assert all(cap==b['cache_gib']*1024 and held+spare<=cap and failed==0 for _,held,spare,cap,_,_,failed in samples)
    direct={'samples':len(samples),'max_held_plus_spare_mib':max((held+spare for _,held,spare,_,_,_,_ in samples),default=0),'evicted':samples[-1][4] if samples else 0,'refused':samples[-1][5] if samples else 0}
    assert all(b['cache_accounting'][k]==v for k,v in direct.items())
    assert 0<=accounting['max_held_plus_spare_mib']<=b['cache_gib']*1024
    assert accounting['samples']>0 if b['cache_gib'] else accounting['samples']==0
    rows=[]
    for c in b['cases']:
        assert len(c['results'])==4
        for stage,v in c['results'].items():
            rows.append((('dialog',c['name'],c['iteration'],stage),v))
            if stage=='prime':assert v['prompt_tokens']==len(r['prompt_ids'][f"{c['name']}-{c['iteration']}"])==targets[c['name']]
            if c['name'].endswith('thinking'):
                assert v['reply']['finish_reason']=='stop'
                assert v['reply']['message']['content'] and v['reply']['message']['reasoning_content']
                counts['completed_thinking']+=1
    assert len(b['shared'])==3
    for i,g in enumerate(b['shared']):
        assert g['peak_running']==4 and len(g['results'])==4
        rows += [(('shared',i,j),v) for j,v in enumerate(g['results'])]
    assert len(b['pressure'])==12
    rows += [(('pressure',i),v['result']) for i,v in enumerate(b['pressure'])]
    c=b['cancel']
    assert c['peak_running']==4 and c['active_before_cancel']==4 and c['active_before_reuse']==3 and c['disconnects_added']==1
    assert len(c['results'])==4
    rows += [(('cancel',name),v) for name,v in c['results'].items()]
    for key,v in rows:
        if b['block']==0:refs[key]=sig(v)
        assert sig(v)==refs[key],(b['block'],key)
        assert type(v['prompt_tokens']) is int and v['prompt_tokens']>0
        assert type(v['tokens']) is int and v['tokens']>0
        assert isinstance(v['token_sha'],str) and re.fullmatch(r'[0-9a-f]{12}',v['token_sha'])
        assert type(v['cached_tokens']) is int and 0<=v['cached_tokens']<v['prompt_tokens']
        assert all(type(v[k]) in (int,float) and math.isfinite(v[k]) and v[k]>=0 for k in ('prefill_seconds','time_to_first_token','seconds','wall_seconds'))
        if b['label']=='off':assert v['cached_tokens']==0
        counts[key[0]]+=1
    delta=b['pressure_cache_after_short_primes']['evicted']-b['pressure_cache_after_long_primes']['evicted']
    assert delta==b['short_prime_evictions']
    assert delta>0 if b['label']=='2k' else delta==0 if b['label'] in ('off','4k') else delta>=0
    if b['label']=='protected':
        assert b['protect_from']==4096
        assert all(x['result']['cached_tokens']>=4096 for x in b['pressure'][8:10])
        assert all(v['cached_tokens']>0 for g in b['shared'][1:] for v in g['results'])
    blocks.append(dict(block=b['block'],label=b['label'],dialog={name:statistics.median(sum(v['wall_seconds'] for v in c['results'].values()) for c in b['cases'] if c['name']==name) for name in sorted(set(c['name'] for c in b['cases']))},shared_first=b['shared'][0]['seconds'],shared_warm=statistics.median(g['seconds'] for g in b['shared'][1:]),pressure_total=sum(v['result']['wall_seconds'] for v in b['pressure']),short_evictions=delta,pressure_revisits=[dict(key=x['key'],wall=x['result']['wall_seconds'],prefill=x['result']['prefill_seconds'],cached=x['result']['cached_tokens']) for x in b['pressure'][8:]],cache=b['cache_accounting']))
assert counts['dialog']==576 and counts['shared']==96 and counts['pressure']==96 and counts['cancel']==32 and counts['completed_thinking']==96
names=sorted({c['name'] for b in r['blocks'] for c in b['cases']})
labels=('off','4k','2k','protected')
dialog={name:{label:statistics.median(sum(v['wall_seconds'] for v in c['results'].values()) for b in r['blocks'] if b['label']==label for c in b['cases'] if c['name']==name) for label in labels} for name in names}
shared={label:statistics.median(g['seconds'] for b in r['blocks'] if b['label']==label for g in b['shared'][1:]) for label in labels}
pressure={label:statistics.median(sum(x['result']['wall_seconds'] for x in b['pressure']) for b in r['blocks'] if b['label']==label) for label in labels}
stages={name:{stage:{label:{metric:statistics.median(c['results'][stage][metric] for b in r['blocks'] if b['label']==label for c in b['cases'] if c['name']==name) for metric in ('prefill_seconds','time_to_first_token','wall_seconds','cached_tokens')} for label in labels} for stage in ('prime','regenerate','follow','edited')} for name in names}
assert dialog==r['dialog_wall_medians']
assert shared==r['shared_warm_medians']
assert pressure==r['pressure_wall_medians']
assert stages==r['medians']
report=dict(exact=True,counts=dict(counts),full_responses=800,aborted_streams=8,blocks=blocks,dialog=dialog,shared=shared,pressure=pressure,stage_medians=stages,limitations=['Three unique dialog fixtures per case, two server starts per mode, six dialog observations','Four warm shared groups per mode; two pressure traces per mode','Shared groups precede all long dialogs in every mode; older receipts used a different order and are not timing controls','First and last cache-off blocks use different unprotected binaries; per-block controls shown','6bit checkpoint, four slots, 24K context, M3 Ultra, RAM cache 2GiB; production 4bit and dual-Mac not tested'])
print(json.dumps(report,indent=2))
