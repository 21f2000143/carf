#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../paper"
echo "Attempting to obtain the official Springer LNCS class from CTAN..."
python - <<'PY'
import urllib.request, zipfile, io, pathlib
url="https://mirrors.ctan.org/macros/latex/contrib/llncs.zip"
data=urllib.request.urlopen(url,timeout=60).read()
z=zipfile.ZipFile(io.BytesIO(data))
for target in ["llncs/llncs.cls","llncs/splncs04.bst","llncs.cls","splncs04.bst"]:
    if target in z.namelist():
        pathlib.Path(pathlib.Path(target).name).write_bytes(z.read(target))
print("Installed llncs.cls / splncs04.bst if present.")
PY
