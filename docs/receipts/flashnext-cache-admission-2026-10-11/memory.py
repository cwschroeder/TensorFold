import collections,json,pathlib,re,sys
all_rows=[json.loads(line) for line in pathlib.Path(sys.argv[1]).read_text().splitlines() if line.strip()]
rows=[r for r in all_rows if r.get('stage')=='serve']
assert all(float(re.search(r'used = ([0-9.]+)M',r['swap_probe']).group(1))==0 for r in all_rows)
ids=list(dict.fromkeys(row['pid'] for row in rows));assert len(ids)==8,ids
blocks=[]
for block,pid in enumerate(ids):
    probes=[row for row in rows if row['pid']==pid]
    swaps=[float(re.search(r'used = ([0-9.]+)M',row['swap_probe']).group(1)) for row in probes]
    blocks.append(dict(block=block,label=['off','4k','2k','protected','protected','2k','4k','off'][block],probes=len(probes),positive_footprint_probes=sum(x['physical_footprint_bytes']>0 for x in probes),zero_footprint_probes=sum(x['physical_footprint_bytes']==0 for x in probes),max_observed_physical_footprint_bytes=max(x['physical_footprint_bytes'] for x in probes),max_observed_physical_footprint_gib=max(x['physical_footprint_bytes'] for x in probes)/2**30,max_swap_used_mib=max(swaps)))
print(json.dumps(dict(total_probes=len(all_rows),checkpoint_probes=sum(r.get('stage')=='checkpoint' for r in all_rows),scope='Physical footprint samples every five seconds, grouped by first-seen native server PID; maximum observed, not a lifetime peak. PIDs and wall timestamps omitted from public summary.',blocks=blocks,all_swap_probes_zero=all(b['max_swap_used_mib']==0 for b in blocks)),indent=2))
