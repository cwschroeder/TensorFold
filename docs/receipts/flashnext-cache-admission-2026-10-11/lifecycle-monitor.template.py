import os
import ctypes,json,pathlib,subprocess,time
root=pathlib.Path(os.environ['TF_RECEIPT_DIR'])
lib=ctypes.CDLL('/usr/lib/libSystem.B.dylib')
lib.proc_pid_rusage.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
lib.proc_pid_rusage.restype=ctypes.c_int
for _ in range(1440):
    try:
        stage='checkpoint' if (root/'checkpoint.pid').exists() else 'serve'
        pid=int((root/('checkpoint.pid' if stage=='checkpoint' else 'serve/server.pid')).read_text())
        info=(ctypes.c_uint64*43)()
        if lib.proc_pid_rusage(pid,4,ctypes.byref(info))==0:
            swap=subprocess.check_output(['/usr/sbin/sysctl','vm.swapusage'],text=True).strip()
            print(json.dumps(dict(time=time.time(),stage=stage,pid=pid,physical_footprint_bytes=info[9],swap_probe=swap)),flush=True)
    except (OSError,ValueError):
        pass
    time.sleep(5)
