#!/usr/bin/env bash
set -euo pipefail

guard="$(dirname "$0")/../scripts/check-linux-abi.sh"
abi_fixture_dir=$(mktemp -d -t radius-abi.XXXXXX)
trap 'rm -f "$abi_fixture_dir/readelf" "$abi_fixture_dir/versions.txt"; rmdir "$abi_fixture_dir"' EXIT
cat > "$abi_fixture_dir/readelf" <<'EOF'
#!/usr/bin/env bash
if [[ -n "${ABI_READELF_FAIL:-}" ]]; then exit 17; fi
cat "$ABI_VERSIONS"
EOF
chmod +x "$abi_fixture_dir/readelf"
export PATH="$abi_fixture_dir:$PATH"
export ABI_VERSIONS="$abi_fixture_dir/versions.txt"

expect_status() {
  local expected=$1
  local actual=0
  local output
  output=$(bash "$guard" fixture 2.35 2>&1) || actual=$?
  if [[ "$actual" != "$expected" ]]; then
    echo "Expected $expected, received $actual: $output" >&2
    exit 1
  fi
}

printf '%s\n' '0x0010: Name: GLIBC_2.35 Flags: none Version: 4' \
  '0x0020: Name: GLIBC_2.43 Flags: WEAK Version: 5' > "$ABI_VERSIONS"
expect_status 0
printf '%s\n' '0x0010: Name: GLIBC_2.43 Flags: none Version: 4' > "$ABI_VERSIONS"
expect_status 1
printf '%s\n' '0x0010: Name: GLIBC_2.9 Flags: none Version: 4' > "$ABI_VERSIONS"
expect_status 0
printf '%s\n' 'No version information found in this file.' > "$ABI_VERSIONS"
expect_status 0
export ABI_READELF_FAIL=1
expect_status 17
echo 'PASS: Linux ABI guard rejects newer strong dependencies and preserves tool failures'
