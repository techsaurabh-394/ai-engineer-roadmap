"""Roadmap text for Phase 8 (fine-tuning, folder 10-fine-tuning). Same shape as the entries in roadmap_data.py."""

PHASE = dict(
    tagline="Change how a model behaves, and prove it helped",
    why="Fine-tuning teaches a model a style, a format or a narrow skill, not new facts. It is the last rung after prompting "
        "and RAG, and it only pays off with clean data, a cheap method such as LoRA, and a fair test against the base model. "
        "This phase shows how to decide, prepare, train and judge.",
    outcomes=["Decide between prompting, RAG and fine-tuning, with a break-even calculation",
              "Prepare a dataset: validate, de-duplicate, scrub, split without leakage, mask the loss",
              "Size a LoRA or QLoRA run and judge the result against the base model on held-out data"],
    ideas=["prompting vs RAG vs tuning", "supervised tuning and loss masks", "dataset preparation", "LoRA", "QLoRA",
           "hyperparameters and loss curves", "evaluation and DPO"],
    builds=["ft_decision.py: when fine-tuning pays for itself", "ft_dataset_prep.py: a leak-free dataset pipeline",
            "ft_lora_params.py: LoRA arithmetic and memory", "ft_eval.py: base vs tuned with confidence intervals"],
    demos=["10-fine-tuning/code/ft_decision.py", "10-fine-tuning/code/ft_dataset_prep.py",
           "10-fine-tuning/code/ft_lora_params.py", "10-fine-tuning/code/ft_eval.py"],
    labs=[],
    needs="Phase 3, especially fine-tuning, quantization and LoRA. Basic Python. No GPU is needed for the code here.",
)
