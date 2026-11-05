#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

binary=${1:?usage: check-linux-abi.sh BINARY MAX_GLIBC}
maximum=${2:?usage: check-linux-abi.sh BINARY MAX_GLIBC}
if [[ ! "$maximum" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]; then
  echo "Invalid GLIBC version: $maximum" >&2
  exit 2
fi

# Weak, dynamically probed libc entry points do not raise the loader baseline.
# Strong libm requirements matter too: v0.3.0 required GLIBC_2.43 there.
required=$(readelf --version-info "$binary" | awk '
  /Name: GLIBC_/ && !/Flags: WEAK/ {
    for (i = 1; i < NF; i++) if ($i == "Name:") print $(i + 1)
  }
' | sed 's/^GLIBC_//' | sort -Vu)
if [[ -z "$required" ]]; then
  echo "$binary: no versioned GLIBC requirements"
  exit 0
fi
highest=$(printf '%s\n' "$required" | tail -n 1)
if [[ "$(printf '%s\n' "$highest" "$maximum" | sort -V | tail -n 1)" != "$maximum" ]]; then
  echo "$binary requires GLIBC_$highest, above the supported GLIBC_$maximum baseline" >&2
  exit 1
fi
echo "$binary: strongest GLIBC requirement $highest (maximum $maximum)"
