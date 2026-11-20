#!/usr/bin/env bash
# Paths: set these for your machine (originals used the author's layout).
: "${PROJECT_DIR:=$(cd "$(dirname "$0")/.." && pwd)}"
: "${LLAMA_DIR:=$HOME/llama.cpp}"; : "${LUCE_DIR:=$HOME/luce}"; : "${DATA:=$HOME}"
# iq2_s ROWS_PER_BLOCK 2 -> 4 AT ncols_dst 3 (our actual B=3 verification width).
#
# WHY THIS IS UNMEASURED: ncols_dst 3..8 all read LUCE_PASCAL_RPB_N4_SUB4 (default 2). The +27%
# iq2_s regression that pinned it at 2 was measured at n=5, and the end-to-end confirmation was
# "prose at B=5 lost 15.8%". Our operating width is n=3, and the in-tree comment states the older
# per-width figures (incl. iq2_s n=3 +2.5%) came from sub-second bursts timing the 405->1189 MHz
# clock ramp and are "unusable in either direction". So n=3 uncapped has never been tested.
#
# WHY IT MATTERS (F140): iq2_s is ~52% of the model and tau is capped by the COST of verification
# width, not by drafter accuracy (acceptance is already 87%). Cheapen the per-column cost and the
# p_min/B optima shift wider, raising tau -- compounding, not additive. Ceiling ~23.6 t/s.
#
# rpb4 = fork rebuilt with -DLUCE_PASCAL_RPB_N4_SUB4=4 (flag verified in flags.make, mmvq.cu
# recompiled, shipped to its own dir so the champion libs are untouched).
set -u
source $PROJECT_DIR/scripts/lib50.sh
cool_cycle "RPB" || exit 1
scp -q /tmp/rpb_arm.sh $H:$HOME/rpb_arm.sh || { echo "ABORT: deploy failed"; exit 1; }
$SSH $H 'chmod +x ~/rpb_arm.sh && echo "rpb_arm deployed after wake"' || exit 1
$SSH $H 'export LD_LIBRARY_PATH=$HOME/rpb4; ldd $HOME/rpb4/llama-server 2>&1 | grep -qi "not found" && { echo "ABORT: rpb4 deps"; exit 1; } || echo "rpb4 deps OK"' || exit 1
$SSH $H "cat > \$HOME/rpb_driver.sh" <<'RMT'
#!/usr/bin/env bash
set -u
O=$HOME/rpb_out.txt; : > "$O"
echo "=== iq2_s rpb 2 vs 4 at ncols_dst 3 $(date +%F_%T) idle=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) ===" >> "$O"
bash $HOME/rpb_arm.sh rpb4   r1 >> "$O" 2>&1
bash $HOME/rpb_arm.sh sub4_2 s1 >> "$O" 2>&1
bash $HOME/rpb_arm.sh rpb4   r2 >> "$O" 2>&1
bash $HOME/rpb_arm.sh sub4_2 s2 >> "$O" 2>&1
echo "RPB_COMPLETE $(date +%F_%T)" >> "$O"
RMT
$SSH $H 'chmod +x $HOME/rpb_driver.sh; setsid nohup bash $HOME/rpb_driver.sh >/dev/null 2>&1 & sleep 2; echo rpb-running'
for i in $(seq 1 150); do $SSH -o ConnectTimeout=8 $H 'grep -q RPB_COMPLETE $HOME/rpb_out.txt' 2>/dev/null && break; sleep 20; done
$SSH $H 'cat $HOME/rpb_out.txt' | tee $PROJECT_DIR/results_RPB.txt
echo "RPB_DONE"
