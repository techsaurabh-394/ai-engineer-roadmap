# Phase 9 · Multimodal AI

**Goal:** understand how AI models read images, audio and video as well as text, so you can choose the right design, estimate its cost, and build a search-and-answer system over mixed content.

**Who it's for:** learners who have finished the LLM basics ([Phase 3](../03-llm-concepts/README.md)) and want to know how pictures and sound get into a model. No signal-processing or computer-vision background is needed.

**Prerequisites:**
- [Phase 3 · LLM Concepts](../03-llm-concepts/README.md), especially [Transformer Architecture](../03-llm-concepts/topics/01-transformer-architecture.md), [Self Attention](../03-llm-concepts/topics/02-self-attention.md) and [Positional Encoding](../03-llm-concepts/topics/04-positional-encoding.md).
- [Phase 1 · RAG Foundations](../01-rag-foundations/README.md) for [embeddings](../01-rag-foundations/topics/02-embeddings.md) and cosine similarity.
- Basic Python and school maths. The code uses only the standard library.

**Time:** about 10-12 hours, including the code.

## Big idea
A transformer only reads **sequences of vectors**. Multimodal AI is the craft of turning each kind of data into such a sequence. An image is cut into **patches**, one vector each. Sound is cut into short **frames** and turned into a spectrogram. Video is a list of sampled frames plus its audio. Once everything is vectors, the same attention machinery applies. Two designs recur: a **bridge** that feeds image tokens into a language model (vision-language models), and a **shared space** where images and text can be compared directly (CLIP). Everything costs tokens, so counting them is a core skill.

## Topics

| # | Topic | Level | Link |
|---|---|---|---|
| 01 | Vision Language Models | Intermediate | [01-vision-language-models.md](topics/01-vision-language-models.md) |
| 02 | How Models See Images | Intermediate | [02-how-models-see-images.md](topics/02-how-models-see-images.md) |
| 03 | Audio and Speech to Text | Intermediate | [03-audio-and-speech-to-text.md](topics/03-audio-and-speech-to-text.md) |
| 04 | Text to Speech | Intermediate | [04-text-to-speech.md](topics/04-text-to-speech.md) |
| 05 | Video Understanding | Advanced | [05-video-understanding.md](topics/05-video-understanding.md) |
| 06 | Multimodal Embeddings and CLIP | Intermediate | [06-multimodal-embeddings-and-clip.md](topics/06-multimodal-embeddings-and-clip.md) |
| 07 | Multimodal RAG | Advanced | [07-multimodal-rag.md](topics/07-multimodal-rag.md) |

## Learning order
1. **Images (02, 01).** Start with [How Models See Images](topics/02-how-models-see-images.md): patches and token counts. Then [Vision Language Models](topics/01-vision-language-models.md) shows how those tokens reach an LLM. (Reading 01 first also works; its token table is explained in 02.)
2. **Sound (03, 04).** Speech to text first, then text to speech, which runs the same pipeline backwards.
3. **Video (05).** Builds on images and audio: frames, sampling and the token bill.
4. **Search across modalities (06, 07).** CLIP gives the shared space; Multimodal RAG puts it to work. It deepens [Multi Modal RAG](../02-advanced-rag/topics/15-multi-modal-rag.md) from Phase 2.

If you are short on time, read 02, 01, 06 and 07.

## Run the code
The demos in `code/` are offline, standard-library Python: no API key, no downloads, no GPU.

```bash
# from the repo root
python 11-multimodal-ai/code/mm_patches.py          # patches, token counts, 2D positions, video token budget
python 11-multimodal-ai/code/mm_audio.py            # sine waves, frames, DFT peak, mel, word error rate
python 11-multimodal-ai/code/mm_clip_contrastive.py # similarity matrix, InfoNCE loss, zero-shot
python 11-multimodal-ai/code/mm_rag.py              # captions + vectors + rank fusion, prompt with image ids

python -m pytest -q 11-multimodal-ai/code           # run the tests (needs: pip install pytest)
```

| File | What it shows | Used in topics |
|---|---|---|
| [`mm_patches.py`](code/mm_patches.py) | cut an image into patches, flatten, count tokens by resolution, position grid, video token budget | 01, 02, 05 |
| [`mm_audio.py`](code/mm_audio.py) | tones, framing, DFT dominant frequency, mel scale, word error rate | 03, 04 |
| [`mm_clip_contrastive.py`](code/mm_clip_contrastive.py) | cosine matrix, symmetric InfoNCE with temperature, zero-shot classification | 06 |
| [`mm_rag.py`](code/mm_rag.py) | BM25 on captions, vector cosine on image features, reciprocal rank fusion, prompt building | 07 |

## Try it
1. In `mm_patches.py`, add resolution 1024 to the table. How many patches and tokens, and how many times more attention pairs than at 224?
2. Change the patch size from 16 to 32. What happens to the token count at 224, and what detail might you lose?
3. In `mm_audio.py`, generate a 150 Hz tone with a 200-sample frame at 800 Hz. Which bin does it land in, and why is the result approximate?
4. Compute the WER of "set a timer for ten minutes" against "set timer for tin minutes".
5. In `mm_clip_contrastive.py`, shuffle the captions so pairs no longer match. How does the loss move compared with the chance value ln(3)?
6. Change the temperature from 0.5 to 0.1 on the toy matrix. Which way does the loss go, and why?
7. In `mm_rag.py`, give `img3` the caption "Quarterly revenue bar chart". Does caption search now find it, and how does the fused order change?
8. In `mm_rag.py`, change the RRF constant `k` from 60 to 1. How much more does a rank-1 result outweigh a rank-3 result?

## Phase checkpoint quiz
Try to answer before opening each answer.

**1. How does a vision transformer turn an image into tokens?**
<details><summary>Answer</summary>
It cuts the image into non-overlapping square patches, flattens each patch into one vector, maps it to the model width with a shared learned linear layer, and adds position information (plus often a CLS token).
</details>

**2. How many tokens does a 224 x 224 image give with 16 x 16 patches, and what happens at 448 x 448?**
<details><summary>Answer</summary>
(224 / 16)^2 = 196 patches (197 with a CLS token). At 448 it is 784 patches: doubling the side quadruples the tokens, and attention work grows about sixteen times.
</details>

**3. What are the three parts of a vision-language model?**
<details><summary>Answer</summary>
An image encoder, a projector (or adapter) that maps image vectors to the language model's embedding width, and the language model itself, which reads image tokens and text tokens together.
</details>

**4. How do LLaVA-style and Flamingo-style designs differ?**
<details><summary>Answer</summary>
LLaVA-style puts projected image tokens into the same sequence as the text, so the LLM is unchanged but images use context. Flamingo-style keeps text as the only sequence and adds cross-attention layers so text attends to image features.
</details>

**5. Why do speech models use a log-mel spectrogram rather than the raw waveform?**
<details><summary>Answer</summary>
It shows which frequencies are present in each short frame, uses mel-spaced bands and a log to match how hearing works, and is much shorter than tens of thousands of raw samples per second.
</details>

**6. How many analysis frames does 1 second of 16 kHz audio give with a 25 ms window and 10 ms hop?**
<details><summary>Answer</summary>
Window 400 samples, hop 160: 1 + (16,000 - 400) // 160 = 98 frames, about 100 per second.
</details>

**7. Reference "the quick brown fox jumps", output "the quick brown box jumps over". What is the word error rate?**
<details><summary>Answer</summary>
One substitution and one insertion over 5 reference words: (1 + 0 + 1) / 5 = 0.40. WER can exceed 1 if the system inserts many words.
</details>

**8. What are the stages of text to speech?**
<details><summary>Answer</summary>
Normalise the text (numbers, dates, symbols), optionally map to phonemes, predict a spectrogram with an acoustic model, then convert it to a waveform with a vocoder. Stream the audio in chunks to reduce time to first audio.
</details>

**9. What is the main safety concern with voice cloning, and what should you do?**
<details><summary>Answer</summary>
Impersonation and fraud. Clone only voices with clear recorded consent, label synthetic audio, protect stored voice samples and have a policy and abuse process.
</details>

**10. A 60-second video is sampled at 1 frame per second at 224 px (197 tokens per frame). What is the token cost, and what if you sample at 2 fps?**
<details><summary>Answer</summary>
60 x 197 = 11,820 tokens at 1 fps, and 23,640 at 2 fps. Cost scales linearly with frames and about four times with doubled frame resolution.
</details>

**11. What do you lose with sparse frame sampling, and how do you reduce the risk?**
<details><summary>Answer</summary>
Events shorter than the gap between samples can be missed and motion is only inferred. Choose the rate from the shortest event that matters, use keyframe or query-aware sampling, and add the audio transcript.
</details>

**12. Explain the CLIP training objective in two sentences.**
<details><summary>Answer</summary>
For a batch of N matched image-text pairs it builds an N x N cosine matrix of normalised embeddings, divides by a temperature, and applies cross-entropy so each image picks its caption and each caption picks its image. The other items in the batch serve as negatives.
</details>

**13. How does zero-shot classification with CLIP work, and what are its limits?**
<details><summary>Answer</summary>
Embed each class as a text prompt and pick the prompt with the highest cosine to the image embedding. It is weak on exact counting, small text and spatial relations, depends on prompt wording, and may fail in domains far from its training data.
</details>

**14. Why index both captions and image embeddings in multimodal RAG?**
<details><summary>Answer</summary>
Captions match exact words but a poor caption such as "Figure 4" hides the image; embeddings find images by content but are weak on exact words and can return look-alikes. Fusing both with reciprocal rank fusion covers each side's gaps.
</details>

**15. How does reciprocal rank fusion score an item, and why use it?**
<details><summary>Answer</summary>
Each list gives an item 1 / (k + rank); the scores are summed across lists. It needs only ranks, so lists with incomparable scores can be merged, and it rewards items several searches agree on. With k = 60, rank 1 in two lists gives 2/61, about 0.033.
</details>

**16. Your multimodal RAG answers chart questions wrongly. Where do you look first?**
<details><summary>Answer</summary>
Retrieval: measure recall@k of the right image on labelled questions. Then check that the actual image (not only the caption) reached the model at a readable size, and that the answer cites ids that exist.
</details>

More practice: [interview prep · LLM fundamentals](../08-interview-prep/01-llm-fundamentals.md).

**Prev:** [Phase 8 · Fine-Tuning](../10-fine-tuning/topics) · **Next:** [Back to the roadmap](../README.md)

---

<p align="center">
  <b>Saurabh Pathak</b> · AI Engineer &amp; Full-Stack Developer<br/>
  <a href="https://github.com/techsaurabh-394">GitHub</a> ·
  <a href="https://www.linkedin.com/in/tech-saurabh/">LinkedIn</a> ·
  <a href="https://mernstackdev.com">Blog</a> ·
  <a href="mailto:saurabhpathak394@gmail.com">saurabhpathak394@gmail.com</a><br/>
  <a href="https://buymeacoffee.com/saurabhdev"><img src="https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black" alt="Buy me a coffee"/></a>
</p>
