from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from enhancer_benchmark_audit.artifacts import materialize, safe_target, sha


class ArtifactIntegrityTests(unittest.TestCase):
    def test_traversal_and_windows_absolute_paths_are_rejected(self):
        for name in ['../outside', '/outside', 'C:/outside', 'a\\..\\outside']:
            with self.assertRaises(ValueError):
                safe_target(REPO / '.runtime/test', name)

    def test_authenticated_archive_and_corrupt_input(self):
        parent = REPO / '.runtime/tests'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            base = Path(folder)
            source = base / 'test.zip'
            content = b'known research input\n'
            with zipfile.ZipFile(source, 'w') as z:
                z.writestr('data/value.txt', content)
            specification = {'key': 'test.zip', 'bytes': source.stat().st_size,
                             'sha256': sha(source), 'members': [{'path': 'data/value.txt',
                             'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}]}
            self.assertEqual(materialize(source, specification, base / 'restored'), 1)
            self.assertEqual((base / 'restored/data/value.txt').read_bytes(), content)
            source.write_bytes(source.read_bytes() + b'corruption')
            with self.assertRaisesRegex(ValueError, 'authentication'):
                materialize(source, specification, base / 'other')
            self.assertFalse((base / 'other').exists())


if __name__ == '__main__':
    unittest.main()
