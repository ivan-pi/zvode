#!/usr/bin/env bash
#
# Reproduce the LFortran 0.63.0 incompatibilities documented in
# docs/lfortran-compatibility.md.
#
# Usage:  LFORTRAN=/path/to/lfortran ./run.sh
# (defaults to `lfortran` on PATH)
#
set -u
LF="${LFORTRAN:-lfortran}"
cd "$(dirname "$0")"

strip() { sed 's/\x1b\[[0-9;]*m//g'; }
firsterr() {
  strip | grep -iE \
    "error:|semantic error|syntax error|tokenizer error|Internal Compiler Error|LCompilersException|code generation error" \
    | head -1
}

echo "LFortran: $($LF --version | head -1)"
echo
echo "== Fixed-form parser (the form zvode.F is written in) =="
echo "   (gfortran -ffixed-form compiles every one of these)"
for f in *_fixed.f; do
  if $LF -c --fixed-form "$f" -o /dev/null >/dev/null 2>&1; then
    printf '  PASS  %s\n' "$f"
  else
    printf '  FAIL  %-34s %s\n' "$f" "$($LF -c --fixed-form "$f" -o /dev/null 2>&1 | firsterr)"
  fi
done

echo
echo "== Free-form counterparts (LFortran's parser accepts all of these) =="
for f in 0[1-6]*_free.f90; do
  if $LF -c "$f" -o /dev/null >/dev/null 2>&1; then
    printf '  PASS  %s\n' "$f"
  else
    printf '  FAIL  %-34s %s\n' "$f" "$($LF -c "$f" -o /dev/null 2>&1 | firsterr)"
  fi
done

echo
echo "== Free-form codegen: a COMMON block shared across module procedures =="
$LF -c 07_common_module_a.f90 -o 07_common_module_a.o >/dev/null 2>&1 \
  && echo "  PASS  07_common_module_a.f90 (compiles to object + .mod)"
if $LF -c 07_common_use_b.f90 -o /dev/null >/dev/null 2>&1; then
  echo "  PASS  07_common_use_b.f90"
else
  printf '  FAIL  %-34s %s\n' "07_common_use_b.f90" \
    "$($LF -c 07_common_use_b.f90 -o /dev/null 2>&1 | firsterr)"
fi
rm -f ./*.o ./*.mod
