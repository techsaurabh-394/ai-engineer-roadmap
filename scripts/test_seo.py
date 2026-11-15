"""SEO and GEO output: text helpers, structured data and the built files (run against a throwaway build)."""
import json
import os
import pathlib
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import seo  # noqa: E402

SITE = "https://example.test/roadmap"


def test_trim_cuts_at_sentence_then_clause_then_word():
    assert seo.trim("One short sentence.", 155) == "One short sentence."
    t = "This first sentence is long enough to count as a real one. " + "word " * 60
    assert seo.trim(t, 80) == "This first sentence is long enough to count as a real one."
    long = "A very long run of words without any stop " * 10
    out = seo.trim(long, 60)
    assert len(out) <= 61 and out.endswith("…") and not out[:-1].endswith(" ")
    assert seo.trim("alpha beta gamma delta epsilon zeta eta theta iota, kappa lambda mu nu xi omicron pi rho", 60).endswith("iota.")


def test_plain_strips_markdown_and_html():
    assert seo.plain("**bold** `code` [link](http://x.y) <b>tag</b> &amp; more") == "bold code link tag & more"


def test_page_title_stays_short():
    assert seo.page_title("Short") == "Short | AI Engineer Roadmap"
    long = "A" * 55
    assert seo.page_title(long) == long


def test_faq_pairs_come_from_the_lessons_own_questions():
    md = "**1. What is X?**\n<details><summary>Answer</summary>\nIt is **Y**.\n</details>\n\n**2. Why?**\n<details><summary>Answer</summary>Because `z`.</details>"
    assert seo.faq_pairs(md) == [("What is X?", "It is Y."), ("Why?", "Because z.")]
    real = (seo.ROOT / "03-llm-concepts/topics/09-temperature.md").read_text(encoding="utf-8")
    assert len(seo.faq_pairs(real)) == 3


def test_every_topic_has_three_faq_pairs():
    bad = [str(p) for p in sorted(seo.ROOT.glob("[01][0-9]-*/topics/*.md")) if len(seo.faq_pairs(p.read_text(encoding="utf-8"))) != 3]
    assert not bad, bad


def test_glossary_terms_parse():
    terms = seo.glossary_terms((seo.ROOT / "docs/glossary.md").read_text(encoding="utf-8"))
    assert len(terms) > 40 and all(t and d for t, d in terms)
    assert any(t.startswith("RAG") for t, _ in terms)


def test_head_meta_needs_a_site_and_escapes():
    assert seo.head_meta("", "x/", "t", "d") == ""
    out = seo.head_meta(SITE, "learn/x/", 'A "quoted" title', "d & e", "article", "2026-01-01", "2026-02-02", "Phase 1")
    assert 'content="A &quot;quoted&quot; title"' in out and "d &amp; e" in out
    assert 'property="article:published_time" content="2026-01-01"' in out and 'name="twitter:card" content="summary_large_image"' in out


def test_article_ld_is_valid_json_with_faq():
    a = dict(title="T", one="One line.", level="Beginner", min=5, phase="01")
    ph = dict(n=1, name="RAG", hub="learn/phase-1/")
    ld = seo.article_ld(SITE, "learn/t/", a, ph, "https://github.com/x/y", "Parent", "https://parent.test", 900, [("Q1?", "A1"), ("Q2?", "A2")], "2026-01-01", "2026-02-02")
    json.dumps(ld)
    types = [x["@type"] for x in ld]
    assert types == ["TechArticle", "BreadcrumbList", "FAQPage"]
    assert ld[0]["dateModified"] == "2026-02-02" and ld[0]["timeRequired"] == "PT5M" and ld[0]["wordCount"] == 900
    assert len(ld[2]["mainEntity"]) == 2


def test_llms_txt_shape():
    txt = seo.llms_txt(SITE, "Summary.", [("Start", [("Home", SITE + "/", "desc")])], [("Code", "https://g.test", "d")])
    lines = txt.splitlines()
    assert lines[0].startswith("# ") and lines[2].startswith("> ")
    assert "- [Home](https://example.test/roadmap/): desc" in txt and "## Optional" in txt


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    import build_site
    out = tmp_path_factory.mktemp("site")
    old = build_site.OUT
    build_site.OUT = out
    env = {"SITE_URL": SITE, "PARENT_SITE_NAME": "Parent", "PARENT_SITE_URL": "https://parent.test", "ADSENSE_CLIENT": "ca-pub-1"}
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        build_site.main()
    finally:
        build_site.OUT = old
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return out


def test_built_pages_have_one_h1_canonical_unique_title_and_valid_jsonld(built):
    titles, descs = {}, {}
    pages = list((built / "learn").glob("*/index.html")) + [built / "learn/index.html", built / "glossary/index.html"]
    assert len(pages) > 130
    for p in pages:
        t = p.read_text(encoding="utf-8")
        rel = p.relative_to(built).parent.as_posix()
        assert t.count("<h1") == 1, rel
        assert re.search(rf'<link rel="canonical" href="{re.escape(SITE)}/{re.escape(rel)}/?">', t), rel
        title = re.search(r"<title>([^<]*)</title>", t).group(1)
        desc = re.search(r'name="description" content="([^"]*)"', t).group(1)
        assert 15 < len(title) <= 80 and 40 <= len(desc) <= 200, (rel, len(title), len(desc))
        assert titles.setdefault(title, rel) == rel, f"duplicate title {title!r}"
        for blob in re.findall(r'<script type="application/ld\+json">(.*?)</script>', t, re.S):
            json.loads(blob.replace("<\\/", "</"))
        assert t.count("adsbygoogle.js") == 1


def test_phase_hubs_glossary_sitemap_robots_llms(built):
    for n in range(0, 10):
        assert (built / f"learn/phase-{n}/index.html").exists(), n
    sm = (built / "sitemap.xml").read_text(encoding="utf-8")
    locs = re.findall(r"<loc>([^<]+)</loc>", sm)
    assert len(locs) == len(set(locs)) > 140 and all(u.startswith(SITE + "/") for u in locs)
    assert f"{SITE}/glossary/" in locs and f"{SITE}/learn/phase-8/" in locs
    for loc in locs:
        sub = loc[len(SITE) + 1:]
        assert (built / sub / "index.html").exists() if sub else (built / "index.html").exists(), loc
    assert f"Sitemap: {SITE}/sitemap.xml" in (built / "robots.txt").read_text(encoding="utf-8")
    llms = (built / "llms.txt").read_text(encoding="utf-8")
    urls = re.findall(r"\]\((https://example\.test/roadmap/[^)]*)\)", llms)
    assert len(urls) > 130
    for u in urls:
        sub = u[len(SITE) + 1:]
        assert (built / sub / "index.html").exists() if sub else (built / "index.html").exists(), u
    assert (built / "llms-full.txt").stat().st_size > 200_000
    assert (built / "img/social-preview.png").exists()


def test_lessons_link_to_each_other_and_to_their_phase_hub(built):
    t = (built / "learn/lora-in-depth/index.html").read_text(encoding="utf-8")
    assert 'href="../phase-8/"' in t and "Related lessons" in t and 'class="answer-first"' in t and "FAQPage" in t


def test_home_page_has_a_crawlable_noscript_list_and_jsonld(built):
    t = (built / "index.html").read_text(encoding="utf-8")
    assert t.count("<h1") == 1 and t.count('<li><a href="learn/') >= 130
    assert '"@type": "Course"' in t and '"@type": "WebSite"' in t and "og:image" in t and t.count("adsbygoogle.js") == 1
