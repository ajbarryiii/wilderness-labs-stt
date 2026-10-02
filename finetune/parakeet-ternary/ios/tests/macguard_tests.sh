# macguard behaviour tests (DESIGN.md "Implementation decisions"). Run with no other guarded job active:
#   MACGUARD_DIR=<dir> sh tests/macguard_tests.sh "<macguard command>" <python for jobs> <scratch dir under the artifacts>
# Prints one line per check; expected exits: 1 0, 2-6 124 (6: second invocation 3, grandchild gone),
# 7 exactly one 0 and one 3, 8 3 and 2, 9 130, 10 3 then cleanup and 0, 11 124 with the job dead,
# 12 3 while the orphaned job lives, 0 after it is killed; 13 yes, 13b /caller/sdk.
# On Linux pass "env MACGUARD_PYTHON=<python> sh ios/macguard" as the macguard command.
G="$1"; PY="$2"; W="$3"; mkdir -p "$W"; A="$MACGUARD_DIR"; LOG="$A/logs/macguard.log"
st() { "$@" 2>/dev/null; echo $?; }
alive() { kill -0 "$1" 2>/dev/null && echo yes || echo no; }
echo "1 harmless: $($G --rss-cap 200M --timeout 30 -- $PY -c "import os; print(os.getpid()==os.getpgrp()==os.getsid(0), os.nice(0), os.environ['OMP_NUM_THREADS'])" 2>/dev/null; echo "exit=$?")"
echo "2 cap: exit=$(st $G --rss-cap 200M --timeout 30 -- $PY -c "import time; b=b'x'*(300<<20); time.sleep(30)") $(grep 'end: status' $LOG | tail -1 | grep -o 'killed: rss [0-9]*KB')"
echo "3 exit 7: exit=$(st $G --rss-cap 200M --timeout 30 -- sh -c 'exit 7')"
echo "4 timeout: exit=$(st $G --rss-cap 200M --timeout 2 -- sleep 30)"
echo "5 TERM handler exits 0: exit=$(st $G --rss-cap 200M --timeout 2 -- $PY -c "import signal,sys,time; signal.signal(signal.SIGTERM, lambda *a: sys.exit(0)); time.sleep(30)")"
P=$W/gc.pid; rm -f $P
( r=$(st $G --rss-cap 200M --timeout 2 -- sh -c "$PY -c 'import os,signal,time; open(\"$P\",\"w\").write(str(os.getpid())); signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)' & sleep 300"); echo "6 TERM-ignoring grandchild: guard exit=$r" ) &
sleep 4; echo "6 second invocation during escalation: exit=$(st $G --rss-cap 200M --timeout 5 -- true)"; wait
echo "6 grandchild alive after: $(alive $(cat $P))"
for i in 1 2; do ( r=$(st $G --rss-cap 200M --timeout 10 -- sh -c "echo RAN$i >> $W/ran; sleep 2"); echo "7 concurrent inv $i exit=$r" ) & done; wait; echo "7 ran: $(cat $W/ran | tr '\n' ' ')"; rm -f $W/ran
echo "8 min-free: exit=$(st $G --min-free 100 --rss-cap 200M --timeout 5 -- true); usage: exit=$(st $G --timeout 5 -- true)"
$G --rss-cap 200M --timeout 60 -- sleep 60 2>/dev/null & m=$!; sleep 2; kill -TERM $m; wait $m; echo "9 supervisor got SIGTERM: exit=$?"
# 10: SIGKILL the supervisor; the sentinel must clean up and keep the lock until then
rm -f $P
$G --rss-cap 1G --timeout 300 -- sh -c "$PY -c 'import os,signal,time; open(\"$P\",\"w\").write(str(os.getpid())); signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)' & sleep 300" 2>/dev/null & m=$!
sleep 3; gc=$(cat $P); kill -KILL $m; sleep 1
echo "10 after supervisor SIGKILL: new invocation exit=$(st $G --rss-cap 200M --timeout 5 -- true) grandchild alive=$(alive $gc)"
sleep 8; echo "10 after sentinel cleanup: grandchild alive=$(alive $gc); new invocation exit=$(st $G --rss-cap 200M --timeout 5 -- true)"
grep "sentinel" $LOG | tail -2 | cut -c1-140
# 11: malformed probe output (valid for the first 2 calls, then malformed)
mkdir -p $W/fakebin; REALPS=$(command -v ps)
printf '#!/bin/sh\nn=$(cat %s/pscount 2>/dev/null || echo 0); echo $((n+1)) > %s/pscount\nif [ "$n" -lt 2 ]; then exec %s "$@"; fi\necho "  123 456 notanumber"\n' $W $W $REALPS > $W/fakebin/ps; chmod +x $W/fakebin/ps; rm -f $W/pscount
rm -f $P; PATH=$W/fakebin:$PATH $G --rss-cap 200M --timeout 60 -- sh -c "echo \$\$ > $P; exec sleep 60" 2>/dev/null; r=$?; sleep 0.5
echo "11 malformed ps: exit=$r job alive=$(alive $(cat $P)) $(grep 'end: status' $LOG | tail -1 | grep -o 'killed: .*' | cut -c1-90)"
rm -f $W/fakebin/ps
if [ "$(uname)" = Darwin ]; then
  printf '#!/bin/sh\nn=$(cat %s/mpcount 2>/dev/null || echo 0); echo $((n+1)) > %s/mpcount\nif [ "$n" -lt 1 ]; then exec /usr/bin/memory_pressure "$@"; fi\necho "System-wide memory free percentage: lots"\n' $W $W > $W/fakebin/memory_pressure; chmod +x $W/fakebin/memory_pressure; rm -f $W/mpcount
  rm -f $P; PATH=$W/fakebin:$PATH $G --rss-cap 200M --timeout 60 -- sh -c "echo \$\$ > $P; exec sleep 60" 2>/dev/null; r=$?; sleep 0.5
  echo "11b malformed memory_pressure: exit=$r job alive=$(alive $(cat $P)) $(grep 'end: status' $LOG | tail -1 | grep -o 'killed: .*' | cut -c1-90)"
  rm -f $W/fakebin/memory_pressure
fi
# 12: supervisor and sentinel both SIGKILLed; the surviving job still holds the inherited lock
rm -f $P; $G --rss-cap 1G --timeout 300 -- sh -c "echo \$\$ > $P; exec sleep 300" 2>/dev/null & m=$!; sleep 2
sen=$(grep "sentinel [0-9]*:" $LOG | tail -1 | sed 's/.*sentinel \([0-9]*\):.*/\1/'); job=$(cat $P)
kill -KILL $m $sen; sleep 1
echo "12 supervisor $m and sentinel $sen killed; job $job alive=$(alive $job); new invocation exit=$(st $G --rss-cap 200M --timeout 5 -- true)"
kill -KILL -- -$job 2>/dev/null; sleep 1; echo "12 after killing the job: job alive=$(alive $job); new invocation exit=$(st $G --rss-cap 200M --timeout 5 -- true)"
# 13: the job's environment equals the caller's (the sh front end restores what the python3 shim injects), apart
# from the thread caps macguard sets on purpose and shell bookkeeping
X='^(SHLVL|_|PWD|OLDPWD|MACGUARD_PYTHON|OMP_NUM_THREADS|VECLIB_MAXIMUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS|NUMEXPR_NUM_THREADS)='
env | grep -v -E "$X" | sort > $W/env.caller
$G --rss-cap 200M --timeout 30 -- sh -c "env | grep -v -E '$X' | sort > $W/env.job" 2>/dev/null
echo "13 job environment == caller environment: $(cmp -s $W/env.caller $W/env.job && echo yes || { echo no:; diff $W/env.caller $W/env.job | head -8 | tr '\n' ' '; })"
SDKROOT=/caller/sdk $G --rss-cap 200M --timeout 30 -- sh -c 'echo "13b caller SDKROOT kept: ${SDKROOT:-unset}"' 2>/dev/null
rm -f $W/env.caller $W/env.job
