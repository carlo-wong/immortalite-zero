"""Download Syzygy 3-4-5 WDL tablebases (~378 MB, 145 .rtbw files)."""
from __future__ import annotations

import argparse
import re
import urllib.parse
import urllib.request
from pathlib import Path

MIRROR = "http://tablebase.sesse.net/syzygy/3-4-5/"
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "syzygy345"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DIR,
        help=f"output directory (default: {DEFAULT_DIR})",
    )
    args = parser.parse_args()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    html = urllib.request.urlopen(MIRROR, timeout=60).read().decode("utf-8", errors="ignore")
    files = sorted(set(re.findall(r'href="([^"/]+\.rtbw)"', html)))
    if not files:
        raise SystemExit(f"No .rtbw files found at {MIRROR}")

    missing = [name for name in files if not (out_dir / name).exists()]
    print(f"Syzygy WDL source: {MIRROR}")
    print(f"Output: {out_dir}")
    print(f"Files: {len(files)} total, {len(missing)} to download")

    for idx, name in enumerate(missing, start=1):
        src = urllib.parse.urljoin(MIRROR, name)
        dst = out_dir / name
        urllib.request.urlretrieve(src, dst)
        if idx % 25 == 0 or idx == len(missing):
            print(f"  Downloaded {idx}/{len(missing)}")

    print(f"Done. {len(list(out_dir.glob('*.rtbw')))} .rtbw files in {out_dir}")


if __name__ == "__main__":
    main()
