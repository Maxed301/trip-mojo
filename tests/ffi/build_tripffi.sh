#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
trip_dir="${TRIP_DIR:-/home/max/Projects/trip_temp}"
out_dir="$repo_dir/build/ffi"

mkdir -p "$out_dir"

inc=(
  "-I$trip_dir"
  "-I$trip_dir/Libraries/MKLibs/U2KX"
  "-I$trip_dir/Libraries/MKLibs/M2KX"
  "-I$trip_dir/Libraries/MKLibs/CMD2KX"
)

gcc -fPIC -ffunction-sections "${inc[@]}" \
  -c "$trip_dir/trprstpnt.c" \
  -o "$out_dir/trprstpnt.o"

gcc -fPIC -ffunction-sections "${inc[@]}" \
  -c "$trip_dir/trpddd.c" \
  -o "$out_dir/trpddd.o"

gcc -fPIC -ffunction-sections "${inc[@]}" \
  -c "$trip_dir/trpdedx.c" \
  -o "$out_dir/trpdedx.o"

gcc -fPIC -ffunction-sections "${inc[@]}" \
  -c "$trip_dir/trpproj.c" \
  -o "$out_dir/trpproj.o"

gcc -fPIC -ffunction-sections "${inc[@]}" \
  -c "$trip_dir/trptarg.c" \
  -o "$out_dir/trptarg.o"

gcc -fPIC -ffunction-sections "${inc[@]}" \
  -c "$trip_dir/Libraries/MKLibs/U2KX/utxt2mem.c" \
  -o "$out_dir/utxt2mem.o"

gcc -fPIC -ffunction-sections "${inc[@]}" \
  -c "$trip_dir/Libraries/MKLibs/U2KX/strnicmp.c" \
  -o "$out_dir/strnicmp.o"

gcc -fPIC -ffunction-sections "${inc[@]}" \
  -c "$trip_dir/Libraries/MKLibs/U2KX/stricnb.c" \
  -o "$out_dir/stricnb.o"

gcc -fPIC -ffunction-sections "${inc[@]}" \
  -c "$trip_dir/Libraries/MKLibs/U2KX/strtrim.c" \
  -o "$out_dir/strtrim.o"

gcc -fPIC -ffunction-sections "${inc[@]}" \
  -c "$trip_dir/Libraries/MKLibs/M2KX/c2d.c" \
  -o "$out_dir/c2d.o"

gcc -fPIC -ffunction-sections "${inc[@]}" \
  -c "$trip_dir/Libraries/MKLibs/U2KX/strnchr.c" \
  -o "$out_dir/strnchr.o"

gcc -fPIC -ffunction-sections "${inc[@]}" \
  -c "$trip_dir/Libraries/MKLibs/M2KX/c2vd.c" \
  -o "$out_dir/c2vd.o"

gcc -fPIC -ffunction-sections "${inc[@]}" \
  -c "$trip_dir/Libraries/MKLibs/U2KX/strnistr.c" \
  -o "$out_dir/strnistr.o"

gcc -fPIC -ffunction-sections "${inc[@]}" \
  -c "$trip_dir/Libraries/MKLibs/M2KX/vlog.c" \
  -o "$out_dir/vlog.o"

gcc -fPIC -ffunction-sections "${inc[@]}" \
  -c "$trip_dir/Libraries/MKLibs/M2KX/splloc.c" \
  -o "$out_dir/splloc.o"

gcc -fPIC -ffunction-sections "${inc[@]}" \
  -c "$trip_dir/Libraries/MKLibs/M2KX/vmov.c" \
  -o "$out_dir/vmov.o"

gcc -fPIC -ffunction-sections "${inc[@]}" \
  -c "$trip_dir/Libraries/MKLibs/M2KX/vintpol1.c" \
  -o "$out_dir/vintpol1.o"

gcc -fPIC "${inc[@]}" \
  -c "$repo_dir/tests/ffi/tripffi_rstpnt.c" \
  -o "$out_dir/tripffi_rstpnt.o"

gcc -shared -Wl,--gc-sections \
  -o "$out_dir/libtripffi.so" \
  "$out_dir/tripffi_rstpnt.o" \
  "$out_dir/trprstpnt.o" \
  "$out_dir/trpddd.o" \
  "$out_dir/trpdedx.o" \
  "$out_dir/trpproj.o" \
  "$out_dir/trptarg.o" \
  "$out_dir/utxt2mem.o" \
  "$out_dir/strnicmp.o" \
  "$out_dir/stricnb.o" \
  "$out_dir/strtrim.o" \
  "$out_dir/c2d.o" \
  "$out_dir/strnchr.o" \
  "$out_dir/c2vd.o" \
  "$out_dir/strnistr.o" \
  "$out_dir/vlog.o" \
  "$out_dir/splloc.o" \
  "$out_dir/vmov.o" \
  "$out_dir/vintpol1.o" \
  -lm
