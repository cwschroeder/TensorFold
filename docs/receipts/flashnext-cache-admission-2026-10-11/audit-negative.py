import copy,gzip,json,pathlib,subprocess,sys,tempfile
root=pathlib.Path(sys.argv[1]);data=json.loads(gzip.decompress((root/'http.json.gz').read_bytes()));assert data['exact']
baseline=subprocess.run([sys.executable,str(root/'analyze.py'),str(root/'http.json.gz')],capture_output=True,text=True)
assert baseline.returncode==0,baseline.stderr
verified=json.loads(baseline.stdout);assert verified['exact'] is True and verified['full_responses']==800
tests=[]
def corrupt(name,mutate,log=False):
 with tempfile.TemporaryDirectory(prefix='tf-admission-audit-') as td:
  dest=pathlib.Path(td)
  for n in range(8):
   src=root/f'serve-block-{n}.log.gz';(dest/src.name).write_bytes(src.read_bytes())
  (dest/'build-pins.json').write_bytes((root/'build-pins.json').read_bytes())
  changed=copy.deepcopy(data)
  mutate(dest if log else changed)
  (dest/'http.json.gz').write_bytes(gzip.compress(json.dumps(changed).encode(),mtime=0))
  p=subprocess.run([sys.executable,str(root/'analyze.py'),str(dest/'http.json.gz')],capture_output=True,text=True)
  assert p.returncode!=0 and 'AssertionError' in p.stderr,(name,p.returncode,p.stderr)
  tests.append(dict(case=name,rejected=True))
def first(d):return next(iter(d['blocks'][3]['cases'][0]['results'].values()))
corrupt('response content changed',lambda d:first(d)['reply']['message'].__setitem__('content','tampered response'))
corrupt('protected long revisit below guard',lambda d:d['blocks'][3]['pressure'][8]['result'].__setitem__('cached_tokens',1))
corrupt('non-finite wall time',lambda d:first(d).__setitem__('wall_seconds',float('nan')))
corrupt('two disconnects instead of one',lambda d:d['blocks'][3]['cancel'].__setitem__('disconnects_added',2))
corrupt('binary digest differs from qualified pins',lambda d:d['binary_sha256'].__setitem__('protected','0'*64))
corrupt('partial receipt marked incomplete',lambda d:d.__setitem__('exact',False))
def bad_log(dest):
 p=dest/'serve-block-3.log.gz';raw=gzip.decompress(p.read_bytes()).decode()
 import re
 raw,n=re.subn(r'(\d+) states, (\d+) MiB and (\d+) MiB spare of (\d+) MiB',lambda m:f'{m[1]} states, 2049 MiB and {m[3]} MiB spare of {m[4]} MiB',raw,count=1)
 assert n==1
 p.write_bytes(gzip.compress(raw.encode(),mtime=0))
corrupt('native log exceeds cache cap',bad_log,True)
print(json.dumps(dict(scope='Deliberately corrupted copies of the completed receipt; no model requests or live changes',baseline_passed=True,cases=tests,all_rejected=True),indent=2))
