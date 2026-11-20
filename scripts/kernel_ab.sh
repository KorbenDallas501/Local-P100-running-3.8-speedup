#!/usr/bin/env bash
# Paths: set these for your machine (originals used the author's layout).
: "${PROJECT_DIR:=$(cd "$(dirname "$0")/.." && pwd)}"
: "${LLAMA_DIR:=$HOME/llama.cpp}"; : "${LUCE_DIR:=$HOME/luce}"; : "${DATA:=$HOME}"
# MASTER vs FORK kernel A/B -- the authoritative measurement of what our Pascal RPB tuning buys.
#
# DISCIPLINE, each item from a failure we already had:
#  * INTERLEAVED, not blocked. master-type1, fork-type1, master-type2, fork-type2, ... Each pair is
#    measured seconds apart at nearly identical temperature, so thermal drift cancels WITHIN the
#    pair instead of being corrected arithmetically afterwards. This is the protocol our own
#    ledger already mandates ("bracket, don't arithmetically correct").
#  * ONE shape only: m=4096 n=1 k=14336, the single case both binaries generate identically
#    (tests/test-backend-ops.cpp:173). --test-file with the real 93-op graph drove the GPU to 80 C
#    in 4 minutes and produced an empty CSV, so it is NOT used here.
#  * >=3 s bursts. Sub-second bursts measure the 405->1189 MHz clock ramp, which is exactly how the
#    retired -25.7%/-15.8%/-13.1% RPB figures were produced. test-backend-ops already self-times to
#    thousands of runs per case, which satisfies this.
#  * Cold start, temp logged before AND after every single case, run aborted if it reaches 78 C.
#  * Both binaries carry sm_60 in the FP16 fast path identically, so that bug does not bias the
#    comparison; it is quantified separately, later, and must not delay this baseline.
set -u
source $PROJECT_DIR/scripts/lib50.sh
FORK=$LUCE_DIR/build/bin
TYPES="q4_1 iq4_xs iq2_s iq3_s q6_K q8_0 f16"

echo "=== ship the FORK test harness to its own dir (never overwrite x50) ==="
cd "$FORK" || { echo "ABORT: no fork build dir"; exit 1; }
[ -x test-backend-ops ] || { echo "ABORT: fork test-backend-ops not built"; exit 1; }
grep -q . /dev/null
tar czf /tmp/forkbin.tgz $(ls *.so*) test-backend-ops || exit 1
until $SSH -o ConnectTimeout=5 $H 'true' 2>/dev/null; do sleep 20; done
$SSH $H 'rm -rf ~/forkbin && mkdir -p ~/forkbin'
scp -q /tmp/forkbin.tgz $H:/tmp/ && $SSH $H 'cd ~/forkbin && tar xzf /tmp/forkbin.tgz && rm -f /tmp/forkbin.tgz && chmod +x test-backend-ops'
# VERIFY the fork binary is the RPB one and x50 is not (F87/F100 lesson: prove the artifact)
$SSH $H 'strings ~/forkbin/libggml-cuda.so.0 | grep -qi "MMVQ_PARAMETERS_PASCAL\|LUCE" && echo "  forkbin: LUCE markers present"'
# the start check must gate the LOCAL driver: a bare `exit 1` inside the ssh quotes only kills the
# REMOTE shell, and the sweep then runs 28 points with a dead arm (which is what happened once).
# this build returns exit 1 from -h while printing usage normally, so gate on OUTPUT, not $?
if $SSH $H 'LD_LIBRARY_PATH=$HOME/forkbin:$HOME/lucefix:$HOME/lucebin ~/forkbin/test-backend-ops -h 2>&1 | grep -q "valid modes"'; then
  echo "  forkbin runs"
else
  echo "ABORT: forkbin will not start -- resolving now:"
  $SSH $H 'LD_LIBRARY_PATH=$HOME/forkbin:$HOME/lucefix:$HOME/lucebin ldd ~/forkbin/test-backend-ops 2>&1 | grep -i "not found"'
  exit 1
fi

$SSH $H "cat > \$HOME/kab.sh" <<'RMT'
#!/usr/bin/env bash
# $1 = tag(master|fork)  $2 = bindir  $3 = type
set -u
T0=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr -dc '0-9')
[ "${T0:-99}" -ge 78 ] && { echo "SKIP $1 $3 -- ${T0}C too hot"; exit 0; }
OUT=$(cd $HOME/$2 && LD_LIBRARY_PATH=$HOME/$2:$HOME/lucefix:$HOME/lucebin timeout 90 ./test-backend-ops perf \
        -o MUL_MAT -p "type_a=$3,type_b=f32,m=4096,n=1,k=14336" 2>/dev/null \
      | sed 's/\x1b\[[0-9;]*m//g' | grep -oE "[0-9]+ runs -[ ]+[0-9.]+ us/run")
T1=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr -dc '0-9')
echo "$1 $3 ${T0}->${T1}C $OUT"
RMT
$SSH $H 'chmod +x $HOME/kab.sh'

cool_cycle "KAB" || exit 1
$SSH $H "cat > \$HOME/kab_driver.sh" <<RMT
#!/usr/bin/env bash
set -u
O=\$HOME/kab_out.txt; : > "\$O"
echo "=== kernel A/B interleaved \$(date +%F_%T) idle=\$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) ===" >> "\$O"
for t in $TYPES; do
  bash \$HOME/kab.sh master x50     \$t >> "\$O" 2>&1
  bash \$HOME/kab.sh fork   forkbin \$t >> "\$O" 2>&1
  bash \$HOME/kab.sh fork   forkbin \$t >> "\$O" 2>&1   # fork twice: replication within the pair
  bash \$HOME/kab.sh master x50     \$t >> "\$O" 2>&1   # and master again = closing bracket per type
done
echo "KAB_COMPLETE \$(date +%F_%T)" >> "\$O"
RMT
$SSH $H 'chmod +x $HOME/kab_driver.sh; setsid nohup bash $HOME/kab_driver.sh >/dev/null 2>&1 & sleep 2; echo kab-running'
for i in $(seq 1 100); do $SSH -o ConnectTimeout=8 $H 'grep -q KAB_COMPLETE $HOME/kab_out.txt' 2>/dev/null && break; sleep 20; done
$SSH $H 'cat $HOME/kab_out.txt' | tee $PROJECT_DIR/results_KAB.txt
echo "KAB_DONE"
