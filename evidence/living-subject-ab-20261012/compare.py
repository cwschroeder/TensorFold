from pathlib import Path
import hashlib,json,re,sys
from lw_project_audit import audit
from lw_project_score import load
p=Path(sys.argv[1])
spec=load(p/'subject_lessons.json',p/'subject_questions.json')
results={}
for name in ('stock','candidate'):
 root=p/name
 data=audit(root,spec)
 (root/'audit.json').write_text(json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
 baseline=json.loads((root/'baseline.json').read_text())['rows']
 last=json.loads((root/'restart.json').read_text())['rows']
 before=json.loads((root/'incident-code.json').read_text())['rows']
 memory=[json.loads(line) for line in (root/'memory.jsonl').read_text().splitlines()]
 swaps=[float(re.search(r'used\s*=\s*([\d.]+)M',r['swap'])[1]) for r in memory if 'swap' in r]
 log=(root/'learning-telemetry.txt').read_text()
 matches=[q['id'] for q in spec['questions'] if q['text'] in log]
 results[name]={'complete':data['complete'],'responses':data['responses'], 'summary':data['summary'],
 'thinking_modes':data['thinking_modes'],'memory':data['memory'],'learning_facts':data['learning_facts'],
 'all_restart_visible_and_finish_equal':all((r['answer'],r['finish'])==(s['answer'],s['finish']) for r,s in zip(before,last)),
 'external_exact_questions_present_in_learner_log':matches,'swap_start_MiB':swaps[0],'swap_max_MiB':max(swaps),
 'project_full_span_diagnostics':sum('neuron key "Project", neighbour subject "Project Aster"' in line for line in log.splitlines()),
 'raw_learning_log_sha256':json.loads((p/'comparison.json').read_text())['runs'][name]['raw_learning_log_sha256']}
 assert len(baseline)==len(last)==len(before)==16
stock=json.loads((p/'stock/baseline.json').read_text())['rows']; candidate=json.loads((p/'candidate/baseline.json').read_text())['rows']
a=json.loads((p/'stock/receipt.json').read_text());b=json.loads((p/'candidate/receipt.json').read_text())
comparison={'same_original_checkpoint':a['source_before']==b['source_before'],'same_corpus':a['hashes']==b['hashes'],
 'baseline_visible_and_finish_equal':all((r['answer'],r['finish'])==(s['answer'],s['finish']) for r,s in zip(stock,candidate)),
 'baseline_message_equal':all(r['response']['choices'][0]['message']==s['response']['choices'][0]['message'] for r,s in zip(stock,candidate)), 'runs':results}
(p/'comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print('paired input/baseline checks:',{k:v for k,v in comparison.items() if k!='runs'})
for name,r in results.items():
 print(name,'lesson',r['summary']['lessons'],'restart',r['summary']['phases'][-1]['groups'],'full-span diagnostics',r['project_full_span_diagnostics'],'frozen exact-question matches',r['external_exact_questions_present_in_learner_log'])
