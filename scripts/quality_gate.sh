#!/usr/bin/env bash
# BROADER QUALITY GATE -- Fable's axis A.
#
# WHY: every closure in this project was validated against ONE deterministic fact (17x23 -> 391).
# That probe earned its keep by flagging F145, but it is a single narrow capability, and the
# ORIGINAL objective was "maximize t/s subject to ACCEPTABLE QUALITY". Nothing has ever checked
# the shipped config for arithmetic beyond one product, instruction-following, or degeneracy.
#
# ALSO: 17x23 has a structural weakness as an A/B instrument. The control already answers it
# CORRECTLY, so any numerics perturbation can only break it -- there is no "toward the reference"
# direction available. It can detect a change; it cannot grade one. Hence a scored set.
#
# All probes are temperature 0 and cache_prompt false => deterministic and directly comparable
# across builds. Scoring is exact-match on a normalised substring, so no judge model is involved.
# $1 = port  $2 = label
set -u
PORT=$1; LABEL=$2
ask(){ curl -s -m 180 http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
   -d "{\"prompt\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$1"),\"n_predict\":$2,\"temperature\":0,\"cache_prompt\":false}" \
   | python3 -c "import json,sys;print(json.load(sys.stdin).get('content','').strip().replace(chr(10),' '))" 2>/dev/null; }

pass=0; tot=0
chk(){ # chk <name> <expected-substring> <n_predict> <prompt>
  local n=$1 exp=$2 np=$3 pr=$4; local got
  got=$(ask "$pr" "$np"); tot=$((tot+1))
  if echo "$got" | grep -qiF "$exp"; then pass=$((pass+1)); echo "  PASS $n"
  else echo "  FAIL $n | want='$exp' got='${got:0:60}'"; fi
}

echo "=== quality gate [$LABEL] $(date +%T) ==="
chk arith1   "391"        16 "Compute 17 times 23 and reply with only the number."
chk arith2   "4896"       16 "Compute 68 times 72 and reply with only the number."
chk arith3   "17"         16 "What is 391 divided by 23? Reply with only the number."
chk multistep "31"        24 "A shop has 48 apples. It sells 17, then receives 12 more, then sells 12. How many apples remain? Reply with only the number."
chk instr1   "BANANA"     16 "Reply with exactly one word, in all capital letters: banana"
chk instr2   "yes"         8 "Answer only yes or no: is 7 a prime number?"
chk instr3   "no"          8 "Answer only yes or no: is 9 a prime number?"
chk fact1    "Paris"      16 "What is the capital of France? Reply with only the city name."
chk fact2    "Jupiter"    16 "Which planet in our solar system is the largest? Reply with only the planet name."
chk code1    "return"     48 "Complete this Python function body in one line. def double(x):"

# DEGENERACY: a longer free generation, scored on distinct-token ratio. A build that has gone
# numerically wrong tends to loop or collapse long before it fails a short factual probe.
LONG=$(ask "Explain in one paragraph why memory bandwidth limits large language model inference speed." 220)
DR=$(python3 -c "
import sys
w=sys.argv[1].split()
print('%.3f'%(len(set(w))/len(w)) if w else '0.000')
print(len(w))" "$LONG" | tr '\n' ' ')
echo "  distinct-ratio+len: $DR"
echo "  long[0:110]: ${LONG:0:110}"
echo "SCORE $LABEL $pass/$tot"
