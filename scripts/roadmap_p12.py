"""Roadmap text for Phase 10 (Frontier AI engineering, folder 12-frontier-ai); imported by roadmap_data.py as PHASES["12"]."""

PHASE = dict(
    tagline="The newer ideas that decide cost, speed and trust",
    why="Models now think before answering, caches decide your bill, small drafts speed up big models, and preference data shapes behaviour. "
        "Teams that understand these levers ship faster and cheaper, and know when a benchmark number can be trusted. "
        "Each lesson turns one idea into arithmetic you can check yourself.",
    outcomes=["Work out when extra thinking or majority voting buys accuracy, and when it makes things worse",
              "Order a prompt so a prefix cache hits, and estimate the saving",
              "Compute speculative-decoding speed-up, DPO loss and pass@k with an error bar by hand"],
    ideas=["reasoning and test-time compute", "context engineering", "prompt caching", "speculative decoding",
           "RLHF and DPO", "distillation", "synthetic data", "pass@k and contamination"],
    builds=["fa_reasoning.py: majority voting maths", "fa_context.py: prefix cache and cost",
            "fa_speculative.py: draft acceptance and speed-up", "fa_preference.py: reward and DPO loss",
            "fa_distill.py: soft targets and a data filter", "fa_evals.py: pass@k, bootstrap, contamination"],
    demos=["12-frontier-ai/code/fa_reasoning.py", "12-frontier-ai/code/fa_context.py", "12-frontier-ai/code/fa_speculative.py",
           "12-frontier-ai/code/fa_preference.py", "12-frontier-ai/code/fa_distill.py", "12-frontier-ai/code/fa_evals.py"],
    labs=[],
    needs="Phases 3 and 8, especially attention, the KV cache and fine-tuning. Basic Python and school maths. No GPU or API key is needed.",
)
