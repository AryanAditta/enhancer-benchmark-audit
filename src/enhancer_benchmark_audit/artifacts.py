"""Authenticate and materialize versioned artifacts without trusting archive paths."""
from pathlib import Path, PurePosixPath
import hashlib
import os
import shutil
import zipfile


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_target(root, relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts or '\\' in relative or any(':' in p for p in rel.parts):
        raise ValueError('Unsafe artifact path')
    root = Path(root).resolve()
    target = root.joinpath(*rel.parts).resolve()
    if root not in target.parents:
        raise ValueError('Artifact path escapes its destination')
    return target


def materialize(source, specification, root):
    """Verify the container first, then every extracted/copied file before use."""
    source = Path(source)
    if source.stat().st_size != specification['bytes'] or sha(source) != specification['sha256']:
        raise ValueError('Container authentication failed: ' + specification['key'])
    members = specification['members']
    if not members:
        return 0
    if specification['key'].endswith('.zip'):
        with zipfile.ZipFile(source) as z:
            listed = [i.filename for i in z.infolist() if not i.is_dir()]
            expected = {r['path']: r for r in members}
            if len(listed) != len(set(listed)) or set(listed) != set(expected):
                raise ValueError('Archive member inventory mismatch')
            targets = {name: safe_target(root, name) for name in listed}
            for name, r in expected.items():
                if z.getinfo(name).file_size != r['bytes']:
                    raise ValueError('Archive member size mismatch')
                target = targets[name]
                if target.exists():
                    if sha(target) != r['sha256']:
                        raise ValueError('Existing artifact differs; preserve it and use a new destination')
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + '.writing')
                with z.open(name) as stream, temporary.open('xb') as output:
                    shutil.copyfileobj(stream, output)
                if sha(temporary) != r['sha256']:
                    raise ValueError('Extracted member authentication failed')
                os.replace(temporary, target)
        return len(members)
    if len(members) != 1 or members[0]['sha256'] != specification['sha256']:
        raise ValueError('Standalone artifact metadata mismatch')
    target = safe_target(root, members[0]['path'])
    if target.exists():
        if sha(target) != specification['sha256']:
            raise ValueError('Existing fitted checkpoint differs; never overwrite')
        return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.writing')
    shutil.copyfile(source, temporary)
    if sha(temporary) != specification['sha256']:
        raise ValueError('Copied artifact authentication failed')
    os.replace(temporary, target)
    return 1
