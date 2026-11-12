"""The roadmap page's text must point at things that exist (demo files, labs, phases, counts)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import roadmap_data as R  # noqa: E402
import site_data  # noqa: E402

ROOT = R.ROOT


def test_every_phase_has_text_for_all_fields():
    assert sorted(R.PHASES) == ["01", "02", "03", "04", "05", "06", "07"]
    for pid, p in R.PHASES.items():
        for key in ("tagline", "why", "outcomes", "ideas", "builds", "demos", "labs", "needs"):
            assert key in p, (pid, key)
        assert p["tagline"] and p["why"] and p["needs"], pid
        assert len(p["outcomes"]) == 3, pid


def test_named_demo_files_exist_and_belong_to_their_phase():
    for pid, p in R.PHASES.items():
        for d in p["demos"]:
            assert (ROOT / d).is_file(), d
            assert d.startswith(pid + "-"), (pid, d)


def test_named_labs_exist_and_are_in_the_right_phase():
    for pid, p in R.PHASES.items():
        for lab in p["labs"]:
            assert lab in site_data.LAB_META, lab
            assert site_data.LAB_META[lab]["phase"] == pid, (pid, lab)


def test_every_lab_is_listed_in_some_phase():
    listed = {lab for p in R.PHASES.values() for lab in p["labs"]}
    assert listed == set(site_data.LAB_META)


def test_after_links_point_at_real_folders():
    for _name, folder, _text in R.AFTER:
        assert (ROOT / folder).is_dir(), folder
