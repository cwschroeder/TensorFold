import os,pathlib,signal,subprocess,sys
root=pathlib.Path(sys.argv[1])
model=pathlib.Path(os.environ['TF_MODEL_DIR'])
os.setsid()
(root/'controller.pid').write_text(str(os.getpid())+'\n')
children=[]
def interrupted(sig,frame):
    raise SystemExit(128+sig)
for sig in (signal.SIGHUP,signal.SIGINT,signal.SIGTERM): signal.signal(sig,interrupted)
def stop(child):
    if child.poll() is None:
        child.terminate()
        try: child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill(); child.wait(timeout=15)
def run(argv,log,pidfile,timeout,name):
    with (root/log).open('w') as out:
        child=subprocess.Popen(argv,stdout=out,stderr=subprocess.STDOUT)
        children.append(child)
        if pidfile: (root/pidfile).write_text(str(child.pid)+'\n')
        try: child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(name+'_TIMEOUT',flush=True)
            raise SystemExit(124)
        finally:
            stop(child)
            if pidfile and child.poll() is not None: (root/pidfile).unlink(missing_ok=True)
        print(name+'_EXIT='+str(child.returncode),flush=True)
        if child.returncode: raise SystemExit(child.returncode)
try:
    fixtures=[root/'prompts'/f'{n}.json' for n in ('short','thinking','sparse','sparse-thinking')]
    run([str(root/'tf-flashnext-batch'),str(model),*map(str,fixtures)],'checkpoint-gate.log','checkpoint.pid',1200,'CHECKPOINT')
    run([sys.executable,str(root/'flashnext_cache_threshold.py'),str(root/'tensorfold-native-4k'),str(root/'tensorfold-native-2k'),str(model),str(root/'serve'),'--rounds','3','--protected',str(root/'tensorfold-native-protected')],'threshold-gate.log',None,5400,'THRESHOLD')
finally:
    for child in reversed(children): stop(child)
    (root/'controller.pid').unlink(missing_ok=True)
