"""Roadmap text for Phase 9 (Multimodal AI); imported by roadmap_data.py as PHASES["11"]."""

PHASE = dict(
    tagline="Teach AI to see, hear and search pictures",
    why="Real data is not only text. Photos, charts, screenshots, speech and video all matter, yet a transformer only reads "
        "sequences of vectors. This phase shows how each kind of data becomes tokens, what that costs, and how to search "
        "and answer across images and text together.",
    outcomes=["Count the tokens an image or a video costs and explain why doubling resolution quadruples them",
              "Explain how speech becomes a spectrogram and score a transcript with word error rate",
              "Work out a CLIP-style contrastive loss and fuse caption and image search into a multimodal RAG prompt"],
    ideas=["vision-language models", "image patches", "audio frames and spectrograms", "speech to text and back",
           "video frame sampling", "CLIP and contrastive loss", "multimodal RAG"],
    builds=["mm_patches.py: patches, token counts, video budget", "mm_audio.py: frames, DFT, word error rate",
            "mm_clip_contrastive.py: similarity matrix and InfoNCE", "mm_rag.py: caption plus vector search with rank fusion"],
    demos=["11-multimodal-ai/code/mm_patches.py", "11-multimodal-ai/code/mm_audio.py",
           "11-multimodal-ai/code/mm_clip_contrastive.py", "11-multimodal-ai/code/mm_rag.py"],
    labs=[],
    needs="Phase 3 (transformers, attention) and the embeddings lesson from Phase 1. Basic Python and school maths.",
)
