#!/usr/bin/env bash
# Apply the 180 W power cap to all Intel Arc Pro B70 (xe) GPUs.
# Invoked at boot by arc-b70-power-caps.service. Idempotent, safe to re-run.
# Matches GPUs by hwmon name == "xe" (hwmon indices are NOT stable across reboots).
set -u

CAP_UW=180000000   # 180 W

# Wait for the xe driver to expose hwmon nodes (up to ~2 min after boot).
found=""
for i in $(seq 1 60); do
  for h in /sys/class/hwmon/hwmon*; do
    [ -e "$h/name" ] || continue
    [ "$(cat "$h/name" 2>/dev/null)" = "xe" ] && { found="$h"; break 2; }
  done
  sleep 2
done
[ -n "$found" ] || { echo "FATAL: no xe hwmon node appeared within 120 s"; exit 1; }

applied=0
for h in /sys/class/hwmon/hwmon*; do
  [ -e "$h/name" ] || continue
  [ "$(cat "$h/name" 2>/dev/null)" = "xe" ] || continue
  [ -e "$h/power1_cap" ] || continue
  echo "$CAP_UW" > "$h/power1_cap"
  echo "set $(basename "$h") -> $(cat "$h/power1_cap") uW"
  applied=$((applied + 1))
done
[ "$applied" -ge 1 ] || { echo "FATAL: no xe GPU exposed power1_cap"; exit 1; }
echo "applied ${CAP_UW} uW cap to $applied xe GPU(s)"
