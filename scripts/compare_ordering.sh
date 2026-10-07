#!/usr/bin/env bash
# Compares the default Morton ordering with min-bbox ordering (--ordering=minbbox) on the same TLR problem:
# off-diagonal tile ranks (after compression and after factorization) and ExaGeoStat's modeling time.
#
# usage: scripts/compare_ordering.sh [ExaGeoStat arguments...]
#   Without arguments a small synthetic TLR problem is used. Any arguments replace the defaults and are passed
#   unchanged to both runs (do not pass --ordering). Real data: --issynthetic=false --datapath=FILE.csv --N=...
#   Environment: EXAGEOSTAT_BIN (example binary, default bin/examples/end-to-end/Example_Data_Generation_and_Modeling),
#                REPS (runs per ordering, default 3; runs alternate between the two orderings).

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${EXAGEOSTAT_BIN:-$ROOT/bin/examples/end-to-end/Example_Data_Generation_and_Modeling}
REPS=${REPS:-3}
if [ ! -x "$BIN" ]; then
  echo "ExaGeoStat example binary not found: $BIN (set EXAGEOSTAT_BIN)" >&2
  exit 1
fi

if [ $# -eq 0 ]; then
  set -- --N=10000 --dts=500 --lts=500 --kernel=UnivariateMaternStationary --dimension=2D --computation=tlr \
         --precision=double --seed=0 --maxrank=200 --accuracy=7 --initial_theta=1:0.1:0.5 \
         --estimated_theta=1:0.1:0.5 --lb=0.05:0.005:0.15 --ub=3:0.5:2 --max_mle_iterations=1
fi

OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

run() { # $1 ordering, $2 repetition, remaining: ExaGeoStat arguments
  local log="$OUT/$1_$2.log"
  "$BIN" "${@:3}" --ordering="$1" > "$log" 2>&1 || { echo "run failed ($1), see output:" >&2; tail -20 "$log" >&2; exit 1; }
  awk -v ordering="$1" '
    /Off-diagonal TLR ranks/ {
      split($0, parts, "|")
      split(parts[2], c, /[ ,]+/); split(parts[3], p, /[ ,]+/)
      for (i in c) { if (c[i] == "avg") cavg = c[i + 1]; if (c[i] == "max") cmax = c[i + 1]; if (c[i] == "squares") csq = c[i + 1] }
      for (i in p) { if (p[i] == "avg") pavg = p[i + 1]; if (p[i] == "max") pmax = p[i + 1]; if (p[i] == "squares") psq = p[i + 1] }
    }
    /LogLi:/ { loglik = $NF }
    /Total Modeling Execution Time/ { total = $NF }
    END { print ordering, total, cavg, cmax, csq, pavg, pmax, psq, loglik }
  ' "$log"
}

for ((r = 0; r < REPS; r++)); do
  if (( r % 2 == 0 )); then order="morton minbbox"; else order="minbbox morton"; fi
  for o in $order; do run "$o" "$r" "$@" >> "$OUT/results.txt"; done
done

awk '
  function median(name, col,    n, i, j, t, v) {
    n = 0
    for (i = 1; i <= rows; i++) if (ord[i] == name) v[++n] = val[i, col]
    for (i = 1; i <= n; i++) for (j = i + 1; j <= n; j++) if (v[j] < v[i]) { t = v[i]; v[i] = v[j]; v[j] = t }
    return n % 2 ? v[(n + 1) / 2] : (v[n / 2] + v[n / 2 + 1]) / 2
  }
  { rows++; ord[rows] = $1; for (c = 2; c <= NF; c++) val[rows, c] = $c }
  END {
    split("modeling time (s)|avg rank, after compression|max rank, after compression|sum of squared ranks, after compression|avg rank, after factorization|max rank, after factorization|sum of squared ranks, after factorization|log-likelihood", label, "|")
    split("%.3f|%.2f|%.0f|%.0f|%.2f|%.0f|%.0f|%.4f", fmt, "|")
    printf "\nMedian over %d runs per ordering (off-diagonal TLR tiles)\n", rows / 2
    printf "%-42s %16s %16s %9s\n", "metric", "morton", "minbbox", "change"
    for (c = 2; c <= 9; c++) {
      m = median("morton", c); b = median("minbbox", c)
      change = (c < 9 && m != 0) ? sprintf("%+.1f%%", (b - m) / m * 100) : ""
      printf "%-42s %16s %16s %9s\n", label[c - 1], sprintf(fmt[c - 1], m), sprintf(fmt[c - 1], b), change
    }
  }
' "$OUT/results.txt"
