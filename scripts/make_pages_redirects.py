"""Turn a built site (_site/) into redirect-only pages for GitHub Pages.

Usage: make_pages_redirects.py SRC_DIR OUT_DIR TARGET_BASE_URL

Every *.html page in SRC_DIR becomes a tiny page in OUT_DIR (same path) that
declares the same path on TARGET_BASE_URL canonical and meta-refreshes to it
instantly. No sitemap, robots or assets are copied, so the mirror cannot compete.
"""
import html
import shutil
import sys
from pathlib import Path


def target_url(base: str, rel: Path) -> str:
    base = base.rstrip("/")
    parts = rel.as_posix()
    if parts == "index.html":
        return base + "/"
    if parts.endswith("/index.html"):
        return f"{base}/{parts[:-len('index.html')]}"
    return f"{base}/{parts}"


def page(url: str) -> str:
    u = html.escape(url, quote=True)
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n'
        f'<title>Redirecting to {u}</title>\n'
        f'<link rel="canonical" href="{u}">\n'
        f'<meta http-equiv="refresh" content="0; url={u}">\n'
        '<meta name="robots" content="noindex,follow">\n'
        f'</head><body><p>Moved to <a href="{u}">{u}</a>.</p>\n'
        f'<script>location.replace({url!r});</script></body></html>\n'
    )


def main(src: str, out: str, base: str) -> int:
    srcp, outp = Path(src), Path(out)
    if outp.exists():
        shutil.rmtree(outp)
    outp.mkdir(parents=True)
    n = 0
    for f in sorted(srcp.rglob("*.html")):
        rel = f.relative_to(srcp)
        dest = outp / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(page(target_url(base, rel)), encoding="utf-8")
        n += 1
    (outp / "404.html").write_text(page(base.rstrip("/") + "/"), encoding="utf-8")
    (outp / ".nojekyll").write_text("")
    print(f"wrote {n} redirect pages to {outp}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
