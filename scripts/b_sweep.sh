#!/usr/bin/env bash
# Paths: set these for your machine (originals used the author's layout).
: "${PROJECT_DIR:=$(cd "$(dirname "$0")/.." && pwd)}"
: "${LLAMA_DIR:=$HOME/llama.cpp}"; : "${LUCE_DIR:=$HOME/luce}"; : "${DATA:=$HOME}"
# B re-optimisation UNDER p_min=0.6 -- the top no-build lever after F127 closed the p_min axis.
#
# WHY: p_min=0.6 cuts MEAN draft width to 1.67 (F104) but we still launch --spec-draft-n-max 3
# (B=4). If the drafter only proposes ~1.7 columns, verification width above that may be paid for
# and unused. This is Fable's "is B=4 still optimal" question reached from the p_min side.
# Also: F22-era B tuning predates BOTH p_min and the surgical requant.
#
# ARMS: n-max 3 (B=4, current), 2 (B=3), 4 (B=5), 5 (B=6), then B=4 again as CLOSING BRACKET.
# A warm probe already EXCLUDED n-max 1 (B=2): 16.43 t/s from 72 C (~16.96 @47C) vs B=4 18.56.
# n-max 1 was +26% WITHOUT p_min; under p_min both are width-reducing and over-narrow. Confirms
# the F124 interaction warning, so that arm is not worth a cold-cycle slot.
# Bracket gate: if the two B=4 arms disagree beyond the thermal slope, interior points are VOID.
# No build -- run_ridge_mtp.sh takes NMAX and PMIN as env vars.
set -u
source $PROJECT_DIR/scripts/lib50.sh
ARMS="3:0.6 2:0.6 4:0.6 5:0.6 3:0.6"

cool_cycle "BSWEEP" || exit 1
scp -q $PROJECT_DIR/scripts/b_arm.sh $H:$HOME/b_arm.sh || { echo "ABORT: arm deploy failed"; exit 1; }
$SSH $H 'chmod +x ~/b_arm.sh && test -x ~/b_arm.sh && echo "b_arm deployed after wake"' || exit 1
$SSH $H "cat > \$HOME/b_driver.sh" <<RMT
#!/usr/bin/env bash
set -u
O=\$HOME/b_out.txt; : > "\$O"
echo "=== B sweep under p_min \$(date +%F_%T) idle=\$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) ===" >> "\$O"
i=0
for a in $ARMS; do
  i=\$((i+1))
  bash \$HOME/b_arm.sh "\${a%%:*}" "\${a##*:}" "\$i" >> "\$O" 2>&1
done
echo "B_COMPLETE \$(date +%F_%T)" >> "\$O"
RMT
$SSH $H 'chmod +x $HOME/b_driver.sh; setsid nohup bash $HOME/b_driver.sh >/dev/null 2>&1 & sleep 2; echo b-running'
for i in $(seq 1 150); do $SSH -o ConnectTimeout=8 $H 'grep -q B_COMPLETE $HOME/b_out.txt' 2>/dev/null && break; sleep 20; done
$SSH $H 'cat $HOME/b_out.txt' | tee $PROJECT_DIR/results_BSWEEP.txt
echo "B_DONE"
