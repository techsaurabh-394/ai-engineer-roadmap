"""Animation content for Phase 9 (multimodal AI)."""
from anim_core import P, L, B  # noqa: F401

# ---------------- Phase 9: multimodal AI ----------------
_T = "11-multimodal-ai/topics/"

P("mm-vision-language-models", "Vision-language model", "An image becomes tokens that sit beside the text tokens.", """
Image|pixels in|1. The picture goes into an image encoder|It was usually trained separately, for example with CLIP.
Features|patch vectors|2. The encoder returns one vector per patch|A grid of vectors, not a sentence.
Project|small adapter|3. A projector maps them into the LLM's token space|Same width as word embeddings.
Join|image + text|4. Image tokens and text tokens form one sequence|The prompt text follows the image tokens.
LLM|reads all|5. The language model attends over both kinds of token|It treats image tokens like extra words.
Answer|text out|6. It writes the answer one token at a time|Cross-attention designs add image links inside the LLM.
""", color="teal", topic=_T + "01-vision-language-models")

P("mm-how-models-see-images", "How a model sees an image", "Cut into patches, flattened, embedded, numbered.", """
Image|224 x 224|1. Start with a grid of pixels|Three numbers per pixel: red, green, blue.
Cut|16 x 16 tiles|2. Cut it into non-overlapping square patches|224 / 16 = 14, so 14 x 14 = 196 patches.
Flatten|768 numbers|3. Each patch becomes one flat vector|16 x 16 x 3 = 768 numbers per patch.
Embed|linear map|4. A learned linear layer maps each to model width|The same layer is used for every patch.
Position|add index|5. Add a position so the grid order is known|Plus one extra CLS token, so 197 tokens.
Encode|transformer|6. Self-attention mixes the patches|Double the side, four times the patches.
""", color="blue", topic=_T + "02-how-models-see-images")

P("mm-audio-and-speech-to-text", "Speech to text", "Sound becomes a picture of frequencies, then words.", """
Wave|16000/s|1. Sound is a list of numbers, 16,000 per second|Each number is the air pressure at one instant.
Frames|25 ms|2. Cut it into short overlapping frames|A 10 ms hop gives 100 frames per second.
Spectrum|frequencies|3. Measure which frequencies are in each frame|This turns time slices into a spectrogram.
Log-mel|80 bands|4. Squash to mel bands and take the log|Closer to how ears hear loudness and pitch.
Encoder|audio model|5. An encoder reads the whole spectrogram|Self-attention over time.
Decoder|writes words|6. A decoder writes text tokens one at a time|Score it with word error rate.
""", color="amber", topic=_T + "03-audio-and-speech-to-text")

P("mm-text-to-speech", "Text to speech", "Clean the text, plan the sound, then draw the wave.", """
Normalise|say it out|1. Turn 12/03 and $5 into spoken words|Numbers, dates and symbols need a reading.
Phonemes|sounds|2. Map words to sounds and stress|Some models skip this and use letters.
Acoustic|to spectrogram|3. An acoustic model predicts a spectrogram|Pitch, length and loudness per frame.
Vocoder|to waveform|4. A vocoder turns the spectrogram into audio|It must make 160 samples per 10 ms frame.
Stream|chunks|5. Send audio in chunks as it is made|The first sound arrives before the sentence ends.
Check|consent|6. Test quality and get consent for any voice clone|Cloning a voice needs permission.
""", color="pink", topic=_T + "04-text-to-speech")

L("mm-video-understanding", "Video understanding", "A video is many images: sample, encode, budget, reason.", """
Sample|1. Pick frames, for example one per second|You cannot afford every frame.
Encode|2. Encode each frame like a normal image|197 tokens each at 224 pixels.
Budget|3. Add up tokens: 60 frames is 11,820 tokens|More fps or pixels costs more.
Order|4. Keep time order with positions or timestamps|Otherwise before and after are lost.
Audio|5. Add the transcript of the audio track|Speech often carries the plot.
Answer|6. The model answers from frames and words|Short events can fall between samples.
""", color="amber", topic=_T + "05-video-understanding")

P("mm-multimodal-embeddings-and-clip", "CLIP: one shared space", "Two encoders learn to put matching pairs close together.", """
Pairs|image + caption|1. Take a batch of matching image and caption pairs|Say 3 pairs: 3 images, 3 captions.
Encode|two encoders|2. One encoder per modality makes a vector each|Then scale every vector to length 1.
Matrix|3 x 3 scores|3. Compare every image with every caption|Cosine similarity gives 9 scores.
Target|the diagonal|4. Matching pairs sit on the diagonal|The other 6 cells are the negatives.
Loss|InfoNCE|5. Softmax each row and column, reward the diagonal|Divide scores by a temperature first.
Use|zero-shot|6. Classify by comparing an image to label prompts|The best-matching prompt wins.
""", color="teal", topic=_T + "06-multimodal-embeddings-and-clip")

P("mm-multimodal-rag", "Multimodal RAG", "Find the right images and text, then show them to the model.", """
Index|caption, vec|1. Store each image with a caption and an embedding|Both routes, so each covers the other's gaps.
Query|question|2. Embed the question as text and as a search vector|One for captions, one for the image space.
Retrieve|two lists|3. Search captions and image vectors separately|Each list can miss different items.
Fuse|RRF|4. Merge the lists with reciprocal rank fusion|Items found by both rise to the top.
Prompt|ids + images|5. Build a prompt that names each image id|Attach the real images, not just captions.
Check|cite and test|6. Ask for cited ids and test retrieval separately|Wrong retrieval looks like a wrong answer.
""", color="blue", topic=_T + "07-multimodal-rag")
