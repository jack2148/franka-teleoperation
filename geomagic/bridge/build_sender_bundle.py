"""Build a tiny stdlib-only zipapp from the actual existing protocol sources."""
from io import BytesIO
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
FILES = (
    "geomagic/__init__.py", "geomagic/bridge/__init__.py",
    "geomagic/bridge/synthetic_sender.py", "geomagic/bridge/timing_stats.py",
    "geomagic/bridge/protocol.py", "geomagic/vendor/__init__.py",
    "geomagic/vendor/geomagic_core/__init__.py", "geomagic/vendor/geomagic_core/state.py",
)


def bundle_bytes():
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        contents = {name: (ROOT / name).read_bytes() for name in FILES}
        contents["__main__.py"] = (
            b"from geomagic.bridge.synthetic_sender import main\nraise SystemExit(main())\n")
        for name, data in contents.items():
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
    return stream.getvalue()


def main():
    destination = ROOT / "geomagic/geomagic_synthetic_sender.pyz"
    with destination.open("xb") as output:
        output.write(bundle_bytes())
    print(destination)


if __name__ == "__main__":
    main()
