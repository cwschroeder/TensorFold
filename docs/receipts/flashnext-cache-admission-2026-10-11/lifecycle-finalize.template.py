import hashlib,json,os,pathlib,re,signal,subprocess,time,urllib.request
root=pathlib.Path(os.environ['TF_RECEIPT_DIR'])
assert (root/'session.exit').read_text().strip()=='0'
r=json.loads((root/'serve/receipt.json').read_text());assert r['exact'] is True and len(r['blocks'])==8
assert 'RESTORED_HEALTH_OK' in (root/'session.log').read_text()
label=os.environ['TF_ORIGINAL_JOB_LABEL']
plist=pathlib.Path.home()/f'Library/LaunchAgents/{label}.plist'
plist_hash=hashlib.sha256(plist.read_bytes()).hexdigest()
assert plist_hash==(root/'original-plist.sha256').read_text().split()[0]
lock=pathlib.Path(os.environ['TF_SESSION_LOCK'])
assert not lock.exists()
def running_owned():
    rows=subprocess.check_output(['ps','-axo','pid=,args='],text=True).splitlines()
    found=[]
    for row in rows:
        pid,_,argv=row.strip().partition(' ')
        if int(pid)==os.getpid():continue
        if str(root) in argv and any(name in argv for name in ('tensorfold-admission-session.sh','controller.py','tf-flashnext-batch','tensorfold-native-','flashnext_cache_threshold.py')):found.append(int(pid))
    return found
for _ in range(15):
    if not running_owned():break
    time.sleep(1)
assert not running_owned(),running_owned()
with urllib.request.urlopen('http://127.0.0.1:52416/health',timeout=4) as f:health=json.load(f)
assert health['status']=='ok' and health['warming'] is False
assert health['model']=='Qwen3.8-Flash-Next-oQ4e-mtp' and health['model_ids']==['Qwen3.8-Flash-Next-oQ4e-mtp','Qwen3.8-Flash-Next-MLX-4bit-MTP']
with urllib.request.urlopen('http://127.0.0.1:52416/v1/metrics',timeout=4) as f:metrics=f.read().decode()
values=dict(re.findall(r'^tensorfold:requests_(running|waiting) (\d+)$',metrics,re.M));assert values=={'running':'0','waiting':'0'},values
port=subprocess.run(['lsof','-n','-iTCP:52417','-sTCP:LISTEN'],capture_output=True,text=True)
assert port.returncode==1 and not port.stdout
subprocess.run(['launchctl','print',f'gui/{os.getuid()}/{label}'],check=True,stdout=subprocess.DEVNULL)
monitorfile=root/'monitor.pid'
if monitorfile.exists():
    pid=int(monitorfile.read_text())
    p=subprocess.run(['ps','-p',str(pid),'-o','args='],capture_output=True,text=True)
    if p.returncode==0:
        assert str(root/'tensorfold-admission-monitor.py') in p.stdout
        os.kill(pid,signal.SIGTERM)
        for _ in range(15):
            q=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True)
            if q.returncode or q.stdout.strip().startswith('Z'):break
            time.sleep(1)
        else:raise RuntimeError('owned monitor did not stop')
monitor_rows=subprocess.check_output(['ps','-axo','stat=,args='],text=True).splitlines()
assert not any(str(root/'tensorfold-admission-monitor.py') in row and not row.strip().startswith('Z') for row in monitor_rows)
life=dict(session_exit=0,restored_healthy=True,restored_health=health,requests_running=0,requests_waiting=0,original_plist_sha256=plist_hash,plist_unchanged=True,benchmark_port_closed=True,owned_benchmark_processes_gone=True,session_lock_removed=True,launchd_job_loaded=True,monitor_stopped=True,candidate_promoted=False,scope='Post-run local checks on M3 Ultra test machine; raw PID and wall-clock evidence retained outside git.')
(root/'final-health.json').write_text(json.dumps(health,indent=2)+'\n')
(root/'final-metrics.txt').write_text(metrics)
(root/'lifecycle.json').write_text(json.dumps(life,indent=2)+'\n')
print(json.dumps(life,indent=2))
