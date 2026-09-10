#!/usr/bin/env bash
set -euo pipefail

paper_dir=$(cd "$(dirname "$0")" && pwd)
project_root=$(dirname "$paper_dir")
archive=${1:-"$project_root/SetHL_ICASSP2027_Overleaf.zip"}

cd "$paper_dir"
"${TECTONIC:-tectonic}" main.tex --keep-logs
python "$project_root/release_audit_content_disjoint.py" --evidence-only

archive=$(realpath -m "$archive")
rm -f "$archive"
files=(main.tex results_content_disjoint.tex refs.bib spconf.sty IEEEbib.bst \
  figures/method_overview.pdf figures/qualitative.pdf)
if command -v zip >/dev/null 2>&1; then
  zip -q "$archive" "${files[@]}"
else
  python - "$archive" "${files[@]}" <<'PY'
import sys
import zipfile

archive, *files = sys.argv[1:]
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
    for path in files:
        output.write(path)
PY
fi
if command -v unzip >/dev/null 2>&1; then
  unzip -tq "$archive"
else
  python -m zipfile --test "$archive"
fi
echo "wrote $archive"
