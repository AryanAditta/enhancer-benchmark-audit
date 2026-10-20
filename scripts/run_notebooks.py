"""Execute exactly the six current notebooks and save executed copies separately."""
from pathlib import Path
import argparse
import json
import os
import time
import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--kernel-name', default='python3')
parser.add_argument('--output', type=Path, default=REPO / 'results/reproduced/executed_notebooks')
args = parser.parse_args()
output = args.output.resolve()
if not output.is_relative_to(REPO) or output.exists():
    raise ValueError('Use a new generated-output directory inside the repository.')
output.mkdir(parents=True)
for key in ['TEMP', 'TMP', 'MPLCONFIGDIR', 'IPYTHONDIR', 'JUPYTER_RUNTIME_DIR', 'JUPYTER_CONFIG_DIR']:
    if key not in os.environ:
        folder = REPO / '.runtime' / key.lower()
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
reports = []
for source in sorted((REPO / 'notebooks').glob('[0-9][0-9]_*.ipynb')):
    started = time.monotonic()
    nb = nbformat.read(source, as_version=4)
    client = NotebookClient(nb, timeout=None, kernel_name=args.kernel_name,
                            resources={'metadata': {'path': str(REPO)}}, allow_errors=False)
    client.execute()
    nbformat.write(nb, output / source.name)
    reports.append({'notebook': source.name, 'passed': True,
                    'seconds': time.monotonic() - started,
                    'code_cells_executed': sum(c.cell_type == 'code' for c in nb.cells)})
    (output / 'progress.json').write_text(json.dumps(reports, indent=2) + '\n')
    print('Completed', source.name, flush=True)
if len(reports) != 6:
    raise ValueError('Expected exactly six current entry notebooks')
(output / 'VALIDATION.json').write_text(json.dumps({'passed': True, 'notebooks': reports,
                                                  'execution': 'Actual Jupyter kernels; original notebook files unchanged.'}, indent=2) + '\n')
