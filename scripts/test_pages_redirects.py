"""The Pages mirror must only redirect to mernstackdev.com and never publish a sitemap."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import make_pages_redirects as m  # noqa: E402

BASE = "https://mernstackdev.com/ai-roadmap"


def test_redirect_pages(tmp_path):
    src, out = tmp_path / "s", tmp_path / "o"
    (src / "learn" / "chunking").mkdir(parents=True)
    (src / "index.html").write_text("<html>real</html>")
    (src / "learn" / "chunking" / "index.html").write_text("<html>real</html>")
    (src / "sitemap.xml").write_text("<urlset/>")
    (src / "robots.txt").write_text("x")
    m.main(str(src), str(out), BASE)
    h = (out / "learn" / "chunking" / "index.html").read_text()
    u = BASE + "/learn/chunking/"
    assert f'rel="canonical" href="{u}"' in h
    assert f'http-equiv="refresh" content="0; url={u}"' in h
    assert BASE + "/" in (out / "index.html").read_text()
    assert not (out / "sitemap.xml").exists() and not (out / "robots.txt").exists()
    assert "real" not in h
