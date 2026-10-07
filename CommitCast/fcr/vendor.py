"""Offline source snapshots, compact enough for one GitHub browser upload."""
from pathlib import Path, PurePosixPath
import shutil
from uuid import uuid4
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"tafas", "petsa", "cosa_current"}


def get_vendor(name):
    if name not in NAMES:
        raise ValueError("Unknown source snapshot")
    # A fully expanded development checkout can be used directly.
    development = ROOT / "baselines/vendor" / name
    if (development / "config.py").is_file():
        return development
    archive = ROOT / "baselines/sources" / (name + ".zip")
    destination = ROOT / ".vendor" / name
    marker = destination / ".extracted"
    with zipfile.ZipFile(archive) as source:
        files = [x for x in source.infolist() if not x.is_dir()]
        for info in files:
            relative = PurePosixPath(info.filename)
            if relative.is_absolute() or '..' in relative.parts or '\\' in info.filename or ':' in info.filename:
                raise ValueError("Unsafe archive path")
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Archive path escapes its destination")
        if marker.is_file() and all((destination/x.filename).is_file() and
                (destination/x.filename).stat().st_size == x.file_size for x in files):
            return destination
        destination.mkdir(parents=True, exist_ok=True)
        for info in files:
            target = destination / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.' + uuid4().hex + '.extracting')
            with source.open(info) as src, temporary.open('wb') as dst:
                shutil.copyfileobj(src,dst)
            temporary.replace(target)
        marker.write_text(f"{len(files)} source files extracted\n", encoding='utf-8')
    return destination
