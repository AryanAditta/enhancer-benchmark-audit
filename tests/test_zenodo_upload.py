"""Offline transport checks; never contact a real deposit."""
from pathlib import Path
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('upload_helper', REPO / 'scripts/upload_to_zenodo.py')
upload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upload)


class ZenodoUploadTests(unittest.TestCase):
    def scenario(self, existing=False, conflicting=False):
        parent = REPO / '.runtime/tests'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            base = Path(folder)
            content = b'authenticated artifact for offline test'
            source = base / 'artifact.dat'
            source.write_bytes(content)
            md5 = hashlib.md5(content).hexdigest()
            plan = base / 'plan.json'
            plan.write_text(json.dumps({'files': [{'key': 'artifact.dat', 'bytes': len(content),
                'sha256': hashlib.sha256(content).hexdigest(), 'local_path': str(source)}]}))
            methods = []

            def fake_urlopen(request, timeout):
                method = request.get_method()
                methods.append(method)
                self.assertTrue(request.full_url.startswith('https://zenodo.org/'))
                if method == 'GET':
                    files = [{'filename': 'artifact.dat', 'checksum': 'wrong' if conflicting else md5}] if existing else []
                    response = {'submitted': False, 'state': 'unsubmitted',
                        'links': {'bucket': 'https://zenodo.org/api/files/offline-only'}, 'files': files}
                else:
                    self.assertEqual(method, 'PUT')
                    self.assertEqual(request.data.read(), content)
                    self.assertEqual(request.get_header('Content-length'), str(len(content)))
                    response = {'checksum': 'md5:' + md5}
                return io.BytesIO(json.dumps(response).encode())

            with patch.object(upload.urllib.request, 'urlopen', side_effect=fake_urlopen), \
                 patch.object(sys, 'argv', ['upload', '--draft-id', '123', '--local-plan', str(plan), '--execute']), \
                 patch.dict(os.environ, {'ZENODO_ACCESS_TOKEN': 'offline-test-value'}), \
                 contextlib.redirect_stdout(io.StringIO()):
                if conflicting:
                    with self.assertRaisesRegex(ValueError, 'not overwritten'):
                        upload.main()
                else:
                    upload.main()
                    result = json.loads((base / 'ZENODO_UPLOAD_PROGRESS.json').read_text())
                    self.assertFalse(result['published'])
                    self.assertEqual(result['completed'][0]['sha256'], hashlib.sha256(content).hexdigest())
            self.assertEqual(methods.count('PUT'), 0 if existing else 1)
            self.assertNotIn('POST', methods)
            self.assertNotIn('DELETE', methods)

    def test_stream_and_verify(self):
        self.scenario()

    def test_matching_upload_is_reused(self):
        self.scenario(existing=True)

    def test_conflicting_upload_is_not_overwritten(self):
        self.scenario(existing=True, conflicting=True)


if __name__ == '__main__':
    unittest.main()
