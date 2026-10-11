#!/bin/bash
set -eu
D="${TF_RECEIPT_DIR:?}"
lock="${TF_SESSION_LOCK:?}"
label="${TF_ORIGINAL_JOB_LABEL:?}"
u=$(id -u)
mkdir "$lock" || exit 2
saved=0
controller=0
cleanup() {
 if [ "$controller" != 0 ]; then
  kill -TERM "$controller" 2>/dev/null || true
  kill -TERM -- "-$controller" 2>/dev/null || true
  for ((i=0;i<15;i++)); do
   kill -0 -- "-$controller" 2>/dev/null || break
   sleep 1
  done
  if kill -0 -- "-$controller" 2>/dev/null; then
   kill -KILL -- "-$controller" 2>/dev/null || true
   sleep 1
  fi
  kill -KILL "$controller" 2>/dev/null || true
  wait "$controller" 2>/dev/null || true
  if kill -0 -- "-$controller" 2>/dev/null; then echo CONTROLLER_GROUP_ALIVE_NO_RESTORE; return 1; fi
 fi
 for pidfile in "$D/checkpoint.pid" "$D/serve/server.pid"; do
  if [ -s "$pidfile" ]; then
   pid=$(cat "$pidfile")
   case "$(ps -p "$pid" -o args=)" in "$D/tf-flashnext-batch "*|"$D/tensorfold-native-4k serve "*|"$D/tensorfold-native-2k serve "*|"$D/tensorfold-native-protected serve "*)
    kill "$pid" 2>/dev/null || true
    sleep 3
    kill -9 "$pid" 2>/dev/null || true
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then echo NATIVE_STILL_ALIVE_NO_RESTORE; return 1; fi;;
   esac
  fi
 done
 ready=1
 if [ "$saved" = 1 ]; then
  ready=0
  launchctl bootstrap "gui/$u" "$HOME/Library/LaunchAgents/$label.plist" || true
  launchctl kickstart "gui/$u/$label" || true
  for ((i=0;i<120;i++)); do
   if curl -fsS --max-time 2 http://127.0.0.1:52416/health >"$D/restored-health.json" 2>/dev/null && python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); sys.exit(not(x.get("status")=="ok" and x.get("warming") is False))' "$D/restored-health.json"; then
    ready=1
    echo RESTORED_HEALTH_OK
    break
   fi
   sleep 1
  done
 fi
 if [ "$ready" != 1 ]; then echo RESTORE_HEALTH_TIMEOUT; return 1; fi
 rmdir "$lock"
}
on_exit() {
 status=$?
 trap - EXIT
 trap "" INT TERM HUP
 if ! cleanup; then status=1; fi
 echo "$status" >"$D/session.exit"
 exit "$status"
}
trap on_exit EXIT
trap 'exit 130' INT TERM HUP
shasum -a 256 "$HOME/Library/LaunchAgents/$label.plist" >"$D/original-plist.sha256"
curl -fsS --max-time 4 http://127.0.0.1:52416/health >"$D/original-health.json"
curl -fsS --max-time 4 http://127.0.0.1:52416/v1/metrics >"$D/drain-metrics.txt"
awk '/^tensorfold:requests_(running|waiting) / {if ($2 != 0) bad=1; n++} END {exit (bad || n!=2)}' "$D/drain-metrics.txt" || { echo BUSY; exit 2; }
launchctl print "gui/$u/$label" >/dev/null
saved=1
launchctl bootout "gui/$u/$label"
sleep 5
if lsof -n -iTCP:52416 -sTCP:LISTEN >/dev/null; then echo SERVER_STILL_LISTENING; exit 2; fi
unset FZ_LANE TF_FLASHNEXT_TP TF_TP_LOCAL FZ_BATCH_MTP_CHAIN
export FZ_N=64
"${TF_PYTHON:?}" "$D/controller.py" "$D" &
controller=$!
wait "$controller"
