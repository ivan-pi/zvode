#!/bin/bash
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Install system dependencies needed to build the Fortran/C extension
apt-get install -y -q gfortran libopenblas-dev

# Install the package with test dependencies
pip install -q -e "${CLAUDE_PROJECT_DIR}[test]"
