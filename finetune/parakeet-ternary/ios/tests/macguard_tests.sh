# macguard behaviour tests (DESIGN.md "Implementation decisions"). Run with no other guarded job active:
#   MACGUARD_DIR=<dir> sh tests/macguard_tests.sh "<macguard command>" <python for jobs> <scratch dir under the artifacts>
# Mac: "<repo>/finetune/parakeet-ternary/ios/macguard"; Linux: "env MACGUARD_PYTHON=<python> sh <repo>/.../ios/macguard".
# Every check prints PASS or FAIL with what it saw; the script exits with the number of failed checks (0 = all pass).
# Fault-injection checks use macguard.py's MACGUARD_TEST_FAULT hook (see its docstring).
G="$1"; PY="$2"; W="$3"; mkdir -p "$W"; A="$MACGUARD_DIR"; LOG="$A/logs/macguard.log"
FAILS=0
st() { "$@" 2>/dev/null; echo $?; }
alive() { kill -0 "$1" 2>/dev/null && echo yes || echo no; }
check() {  # check NAME EXPECTED ACTUAL
  if [ "$2" = "$3" ]; then echo "PASS $1: $3"; else echo "FAIL $1: expected [$2] got [$3]"; FAILS=$((FAILS + 1)); fi
}
lastend() { grep -E 'end: (status|finalization)' $LOG | tail -1; }
waitfree() {  # wait up to $1 s until a new invocation can take the lock
  i=0; while [ $i -lt "$1" ]; do [ "$(st $G --rss-cap 200M --timeout 5 -- true)" = 0 ] && return 0; sleep 1; i=$((i + 1)); done; return 1
}

o=$($G --rss-cap 200M --timeout 30 -- $PY -c "import os; print(os.getpid()==os.getpgrp()==os.getsid(0), os.nice(0), os.environ['OMP_NUM_THREADS'])" 2>/dev/null); r=$?
check "1 harmless (own group, nice 10, threads 4)" "True 10 4 exit=0" "$o exit=$r"
check "2 rss cap" "124 killed: rss" "$(st $G --rss-cap 200M --timeout 30 -- $PY -c "import time; b=b'x'*(300<<20); time.sleep(30)") $(lastend | grep -o 'killed: rss')"
check "3 job status passes through" 7 "$(st $G --rss-cap 200M --timeout 30 -- sh -c 'exit 7')"
check "4 timeout" 124 "$(st $G --rss-cap 200M --timeout 2 -- sleep 30)"
check "5 TERM handler exiting 0 is still an abort" 124 "$(st $G --rss-cap 200M --timeout 2 -- $PY -c "import signal,sys,time; signal.signal(signal.SIGTERM, lambda *a: sys.exit(0)); time.sleep(30)")"
P=$W/gc.pid; rm -f $P
( r=$(st $G --rss-cap 200M --timeout 2 -- sh -c "$PY -c 'import os,signal,time; open(\"$P\",\"w\").write(str(os.getpid())); signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)' & sleep 300"); echo "$r" > $W/r6 ) &
sleep 4; check "6 second invocation during escalation refused" 3 "$(st $G --rss-cap 200M --timeout 5 -- true)"; wait
check "6 TERM-ignoring grandchild: guard aborts, grandchild gone" "124 no" "$(cat $W/r6) $(alive $(cat $P))"
rm -f $W/ran $W/r7.*
for i in 1 2; do ( st $G --rss-cap 200M --timeout 10 -- sh -c "echo RAN$i >> $W/ran; sleep 2" > $W/r7.$i ) & done; wait
check "7 concurrent invocations: one runs, one refused" "0 3 1" "$(cat $W/r7.1 $W/r7.2 | sort | tr '\n' ' ')$(wc -l < $W/ran | tr -d ' ')"
check "8 min-free refusal, usage error" "3 2" "$(st $G --min-free 100 --rss-cap 200M --timeout 5 -- true) $(st $G --timeout 5 -- true)"
$G --rss-cap 200M --timeout 60 -- sleep 60 2>/dev/null & m=$!; sleep 2; kill -TERM $m; wait $m
check "9 supervisor got SIGTERM" 130 "$?"
# 10: SIGKILL the supervisor; the sentinel must clean up and keep the lock until then
rm -f $P
$G --rss-cap 1G --timeout 300 -- sh -c "$PY -c 'import os,signal,time; open(\"$P\",\"w\").write(str(os.getpid())); signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)' & sleep 300" 2>/dev/null & m=$!
sleep 3; gc=$(cat $P); kill -KILL $m; sleep 1
check "10 supervisor SIGKILLed: lock kept while the grandchild lives" "3 yes" "$(st $G --rss-cap 200M --timeout 5 -- true) $(alive $gc)"
sleep 8
check "10 after the sentinel's cleanup: grandchild gone, lock free" "no 0" "$(alive $gc) $(st $G --rss-cap 200M --timeout 5 -- true)"
# 11: malformed probe output (valid for the first 2 calls, then malformed)
mkdir -p $W/fakebin; REALPS=$(command -v ps)
printf '#!/bin/sh\nn=$(cat %s/pscount 2>/dev/null || echo 0); echo $((n+1)) > %s/pscount\nif [ "$n" -lt 2 ]; then exec %s "$@"; fi\necho "  123 456 notanumber"\n' $W $W $REALPS > $W/fakebin/ps; chmod +x $W/fakebin/ps; rm -f $W/pscount
rm -f $P; PATH=$W/fakebin:$PATH $G --rss-cap 200M --timeout 60 -- sh -c "echo \$\$ > $P; exec sleep 60" 2>/dev/null; r=$?; sleep 0.5
check "11 malformed ps aborts, job dead" "124 no malformed ps row" "$r $(alive $(cat $P)) $(lastend | grep -o 'malformed ps row')"
rm -f $W/fakebin/ps
if [ "$(uname)" = Darwin ]; then
  printf '#!/bin/sh\nn=$(cat %s/mpcount 2>/dev/null || echo 0); echo $((n+1)) > %s/mpcount\nif [ "$n" -lt 2 ]; then exec /usr/bin/memory_pressure "$@"; fi\necho "System-wide memory free percentage: lots"\n' $W $W > $W/fakebin/memory_pressure; chmod +x $W/fakebin/memory_pressure; rm -f $W/mpcount
  rm -f $P; PATH=$W/fakebin:$PATH $G --rss-cap 200M --timeout 60 -- sh -c "echo \$\$ > $P; exec sleep 60" 2>/dev/null; r=$?; sleep 0.5
  check "11b malformed memory_pressure aborts, job dead" "124 no" "$r $(alive $(cat $P 2>/dev/null || echo 0))"
  rm -f $W/fakebin/memory_pressure
fi
# 12: supervisor and sentinel both SIGKILLed; the surviving job still holds the inherited lock (residual risk:
# a job that closed it would not; see macguard.py)
rm -f $P; $G --rss-cap 1G --timeout 300 -- sh -c "echo \$\$ > $P; exec sleep 300" 2>/dev/null & m=$!; sleep 2
sen=$(grep "sentinel [0-9]*:" $LOG | tail -1 | sed 's/.*sentinel \([0-9]*\):.*/\1/'); job=$(cat $P)
kill -KILL $m $sen; sleep 1
check "12 supervisor and sentinel SIGKILLed: job alive, holds the lock" "yes 3" "$(alive $job) $(st $G --rss-cap 200M --timeout 5 -- true)"
kill -KILL -- -$job 2>/dev/null; sleep 1
check "12 after killing the job: lock free" "no 0" "$(alive $job) $(st $G --rss-cap 200M --timeout 5 -- true)"
# 13: the job's environment equals the caller's (the sh front end restores what the python3 shim injects), apart
# from the thread caps macguard sets on purpose and shell bookkeeping
X='^(SHLVL|_|PWD|OLDPWD|MACGUARD_PYTHON|OMP_NUM_THREADS|VECLIB_MAXIMUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS|NUMEXPR_NUM_THREADS)='
env | grep -v -E "$X" | sort > $W/env.caller
$G --rss-cap 200M --timeout 30 -- sh -c "env | grep -v -E '$X' | sort > $W/env.job" 2>/dev/null
check "13 job environment == caller environment" "same" "$(cmp -s $W/env.caller $W/env.job && echo same || diff $W/env.caller $W/env.job | head -6 | tr '\n' ' ')"
check "13b caller SDKROOT kept" "/caller/sdk" "$(SDKROOT=/caller/sdk $G --rss-cap 200M --timeout 30 -- sh -c 'echo ${SDKROOT:-unset}' 2>/dev/null)"
rm -f $W/env.caller $W/env.job
# 14: sentinel-only SIGKILL: the supervisor notices, kills the job and exits 124 (never 0)
rm -f $P; $G --rss-cap 1G --timeout 300 -- sh -c "echo \$\$ > $P; exec sleep 300" 2>/dev/null & m=$!; sleep 2
sen=$(grep "sentinel [0-9]*:" $LOG | tail -1 | sed 's/.*sentinel \([0-9]*\):.*/\1/'); job=$(cat $P)
kill -KILL $sen; wait $m; r=$?
check "14 sentinel SIGKILLed: abort, job dead, lock free" "124 no sentinel 0" "$r $(alive $job) $(lastend | grep -o 'sentinel' | head -1) $(st $G --rss-cap 200M --timeout 5 -- true)"
# 15: every signal raises PermissionError (supervisor and sentinel): not verifiable, exit 125, the lock stays held
# until the group is really gone (killed here by the test), then it is released
rm -f $P; env MACGUARD_TEST_FAULT=killpg-eperm $G --rss-cap 200M --timeout 2 -- sh -c "echo \$\$ > $P; exec sleep 300" 2>/dev/null; r=$?
job=$(cat $P)
check "15 cleanup PermissionError: 125, job alive, lock retained" "125 yes 3" "$r $(alive $job) $(st $G --rss-cap 200M --timeout 5 -- true)"
kill -KILL -- -$job 2>/dev/null; sleep 2
check "15 after the group is gone: lock released" "no 0" "$(alive $job) $(waitfree 10 >/dev/null; st $G --rss-cap 200M --timeout 5 -- true)"
# 16: supervisor dies between starting the gated wrapper and the acknowledgement: wrapper and job never run CMD
rm -f $W/ran16
env MACGUARD_TEST_FAULT=die-before-ack $G --rss-cap 200M --timeout 30 -- sh -c "echo RAN > $W/ran16; sleep 3016" 2>/dev/null; r=$?
sleep 3
check "16 supervisor dead before ACK: killed, CMD never ran, no wrapper left, lock free" "137 no 0 0" \
  "$r $(test -e $W/ran16 && echo yes || echo no) $(ps -A -o command= | grep -c '[e]xec-after-gate.*sleep 3016') $(waitfree 10 >/dev/null; st $G --rss-cap 200M --timeout 5 -- true)"
# 17: a failing end log turns a successful job into 125 (fail closed)
check "17 end-log failure: 125" 125 "$(st env MACGUARD_TEST_FAULT=end-log-fail $G --rss-cap 200M --timeout 30 -- true)"
rm -f $W/r6 $W/r7.* $W/ran $W/ran16
echo "failed checks: $FAILS"
exit $FAILS
