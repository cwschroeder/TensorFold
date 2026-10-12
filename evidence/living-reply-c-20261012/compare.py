"""Audit a completed same-binary policy pair, with frozen inputs and owned restarts."""
from pathlib import Path
import hashlib,json,re,sys
from lw_project_audit import audit
from lw_project_score import load
base,root=map(Path,sys.argv[1:3])
spec=load(root/'subject_lessons.json',root/'subject_questions.json')
before=base/'candidate';after=root/'reply'
b=json.loads((before/'receipt.json').read_text());c=json.loads((after/'receipt.json').read_text())
assert b['binary_sha256']==c['binary_sha256']
assert b['hashes']==c['hashes']==spec['hashes']
assert b['source_before']==c['source_before']
assert c['build']['learning_environment']=={'TF_SLIDE_POCKET_KEY':'reply'}
results={}
logs={}
for name,p in [('auto_subject',before),('reply',after)]:
 data=audit(p,spec)
 if name=='reply':(p/'audit.json').write_text(json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
 def phase(n):return json.loads((p/(n+'.json')).read_text())['rows']
 initial=phase('baseline'); learned=phase('incident-code'); restarted=phase('restart')
 log=(p/'learning-telemetry.txt').read_text();logs[name]=log.splitlines()
 assert len(initial)==len(learned)==len(restarted)==16
 matches=[q['id'] for q in spec['questions'] if q['text'] in log]
 assert matches==[]
 memory=[json.loads(x) for x in (p/'memory.jsonl').read_text().splitlines()]
 swaps=[float(re.search(r'used\s*=\s*([\d.]+)M',r['swap'])[1]) for r in memory if 'swap' in r]
 results[name]={'complete':data['complete'],'responses':data['responses'],'summary':data['summary'],
 'thinking_modes':data['thinking_modes'],'memory':data['memory'],'learning_facts':data['learning_facts'],
 'all_restart_visible_and_finish_equal':all((x['answer'],x['finish'])==(y['answer'],y['finish']) for x,y in zip(learned,restarted)),
 'exact_external_questions_in_learner_log':matches,
 'neuron_allocation_lines':[x for x in logs[name] if "the fact's neurons:" in x],
 'swap_start_MiB':swaps[0],'swap_max_MiB':max(swaps),
 'raw_learning_log_sha256':json.loads((root/'comparison.json').read_text())['runs'][name]['raw_learning_log_sha256']}
initial_b=json.loads((before/'baseline.json').read_text())['rows']
initial_c=json.loads((after/'baseline.json').read_text())['rows']
markers={'retained_subject_questions':'neuron key','near_misses':'near misses kept steady:',
 'thinking_training':'questions also learned with thinking on','router_counts':'its router also reads'}
preparation={}
for label,mark in markers.items():
 x=[line for line in logs['auto_subject'] if mark in line];y=[line for line in logs['reply'] if mark in line]
 preparation[label]={'equal':x==y,'before_lines':len(x),'reply_lines':len(y)}
comparison={'same_binary':True,'same_corpus':True,'same_original_checkpoint':True,
 'baseline_full_messages_equal':all(x['response']['choices'][0]['message']==y['response']['choices'][0]['message'] for x,y in zip(initial_b,initial_c)),
 'baseline_visible_and_finish_equal':all((x['answer'],x['finish'])==(y['answer'],y['finish']) for x,y in zip(initial_b,initial_c)),
 'observed_preparation':preparation,'runs':results}
assert comparison['baseline_full_messages_equal'] and comparison['baseline_visible_and_finish_equal']
assert all(x['equal'] for x in preparation.values())
assert all(r['all_restart_visible_and_finish_equal'] for r in results.values())
assert all('10 at the reply' in x and "0 at its questions' subject" in x for x in results['reply']['neuron_allocation_lines'])
assert results['reply']['neuron_allocation_lines']
final=json.loads((after/'restart.json').read_text())['rows']
comparison['restart_full_messages_equal_original_baseline']=all(x['response']['choices'][0]['message']==y['response']['choices'][0]['message'] for x,y in zip(initial_b,final))
assert comparison['restart_full_messages_equal_original_baseline']
(root/'comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
for name,data in results.items():
 print(name,data['summary']['lessons'],data['summary']['phases'][-1]['groups'])
print('verified equal inputs, initial full messages, observed preparation and genuine restart')
