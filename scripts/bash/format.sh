#!/bin/bash

: ${CFM:=clang-format}
: ${PFM:=ruff}
: ${LFM:=stylua}
: ${VERBOSE:=0}

if ! command -v ${CFM} &> /dev/null; then
    >&2 echo "[format.sh] Error: No clang format found! Looked for ${CFM}"
    exit 1
else
    CFM=$(command -v ${CFM})
    echo "[format.sh] Clang format found: ${CFM}"
fi

# clang format major version
TARGET_CF_VRSN=20.1.0
CF_VRSN=$(${CFM} --version)
echo "[format.sh] Note we assume clang format version ${TARGET_CF_VRSN}."
echo "[format.sh] You are using ${CF_VRSN}."
echo "[format.sh] If these differ, results may not be stable."

echo "[format.sh] Formatting..."
REPO=$(git rev-parse --show-toplevel)
for f in $(git grep --untracked -ail ';' -- ':/*.hpp' ':/*.cpp'); do
    if [ ${VERBOSE} -ge 1 ]; then
       echo ${f}
    fi
    ${CFM} -i ${f}
done

# format python files
if ! command -v ${PFM} &> /dev/null; then
    >&2 echo "[format.sh] Error: No version of ruff found! Looked for ${PFM}"
    exit 1
else
    PFM=$(command -v ${PFM})
    echo "[format.sh] ruff Python formatter found: ${PFM}"
    echo "[format.sh] ruff version: $(${PFM} --version)"
fi

echo "[format.sh] Formatting Python files..."
REPO=$(git rev-parse --show-toplevel)
for f in $(git grep --untracked -ail res -- :/*.py); do
    if [ ${VERBOSE} -ge 1 ]; then
       echo ${f}
    fi
    ${PFM} format ${f}
done

# format lua files
if ! command -v ${LFM} &> /dev/null; then
    >&2 echo "[format.sh] Error: No stylua found! Looked for ${LFM}"
    exit 1
else
    LFM=$(command -v ${LFM})
    echo "[format.sh] stylua found: ${LFM}"
    echo "[format.sh] stylua version: $(${LFM} --version)"
fi

echo "[format.sh] Formatting Lua files..."
for f in $(git ls-files '*.lua'); do
    if [ ${VERBOSE} -ge 1 ]; then
        echo ${f}
    fi
    ${LFM} ${f}
done

echo "[format.sh] ...Done"
