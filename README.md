# SpeechPipeline

A Python pipeline that uses YouTube audio, figures out who said what and when, tags each speech segment with an emotion, and produces clean structured data ready for training expressive text-to-speech models. 
It also has a data governance layer that checks every source's licence before touching it, so nothing proprietary quietly sneaks into the training set.

## Key Objectives

- Automating the full journey from a YouTube URL to annotated training data
- Making sure every audio source is licence-checked before any processing happens
- Handling the messy things like bad formats, short segments, model uncertainty, failed downloads
- Producing outputs (JSON annotations, RTTM files, CSV manifests) that downstream models can consume directly
- Making the pipeline safe to run unattended on hundreds of files 

## Methodology

The pipeline runs in six sequential stages. Each stage is its own module, has its own error class, and fails loudly rather than silently passing bad data forward.

1. Download audio using `yt-dlp` and capture full source metadata
2. Check the licence before doing anything else block non-open sources at the gate
3. Convert to 16kHz mono WAV using `FFmpeg` and validate the output
4. Run speaker diarization to get timestamped speaker segments
5. Tag each segment with an emotion label and confidence score
6. Assemble everything into a final annotation JSON with provenance embedded

The pipeline is idempotent .i.e. if you run it twice on the same URL, it skips files that are already processed. 
This matters when you're running on thousands of files and something crashes halfway through.


## Model Pipeline

**Speaker Diarization — `pyannote/speaker-diarization-3.1`**

This model answers the question "who spoke when". It segments the audio into turns and assigns a speaker label to each one. The output is saved as both an RTTM file (the standard diarization format used in benchmarking) and a structured JSON with start time, end time, speaker ID, and duration for every segment.

After diarization runs, a validation step checks for red flags: only one speaker detected, too many speakers, very low speech ratio, or excessive short-segment fragmentation. These get flagged rather than silently passed downstream.

**Emotion Recognition — `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`**

This model classifies each speech segment into one of seven emotion categories: angry, disgusted, fearful, happy, neutral, sad, surprised. It runs on the audio slice for each diarization segment individually.

Every prediction comes with a confidence score. Segments that score below 0.6 are flagged as `LOW_CONFIDENCE` rather than accepted as ground truth. This is important — bad emotion labels in training data produce bad TTS models.

## Safety

The most important safety feature in this pipeline is the consent gate, and it runs before any ML model touches the audio.

Every source gets classified into one of four licence tiers:

| Tier | What it means | Decision |
|---|---|---|
| OPEN | Creative Commons BY, CC0, reuse explicitly allowed | Processed |
| RESTRICTED | CC BY-NC, CC BY-ND | Blocked |
| PROPRIETARY | YouTube Standard Licence | Blocked |
| UNKNOWN | No licence info found | Blocked |

If a source doesn't pass, the pipeline raises a `ConsentError` and moves on. The audio file gets downloaded (to capture the metadata), but it never reaches diarization or emotion tagging. 
Every decision is logged and written to a `provenance_manifest.csv` so you have a full audit trail of what was allowed and what was blocked.

The second safety layer is the emotion confidence gate. A segment tagged with 13% confidence isn't a reliable annotation, it's noise. 
These get flagged as `LOW_CONFIDENCE` and can be filtered out before training rather than silently degrading model quality.

## Edge Cases Handled

- **Failed downloads** — retry logic with configurable max attempts, raises `DownloadError` after exhausting retries
- **Format mismatches** — FFmpeg converts any input format (webm, mp4, m4a, mp3) to 16kHz mono WAV
- **Wrong sample rate or channel count** — post-conversion validation catches this before diarization
- **Audio too short** — files under 1 second are rejected at validation
- **Short diarization segments** — segments under 1 second are skipped for emotion tagging (too little audio for a reliable prediction)
- **Model errors during emotion tagging** — caught per-segment, logged, flagged as `MODEL_ERROR`, pipeline continues
- **Missing licence metadata** — treated as UNKNOWN tier, blocked
- **Already processed files** — idempotency check skips them, no redundant computation
- **Single-speaker recordings** — diarization validation flags this as a potential issue without blocking

## Challenges

The biggest technical challenge was dependency compatibility. `pyannote.audio 3.3.2` conflicted with newer versions of `torch`, `torchaudio`, `huggingface_hub`, `numpy`, and `speechbrain` in ways that weren't obvious upfront. Resolving this required:

- Downgrading to Python 3.11 (the pipeline was initially built on 3.14 where pyannote fails to import)
- Pinning `torch==2.3.0` and `torchaudio==2.3.0` to avoid a missing `fbgemm.dll` dependency on Windows
- Downgrading `speechbrain` to `0.5.16` to avoid a missing `k2` module
- Pinning `huggingface_hub==0.23.4` to match pyannote's internal `use_auth_token` API

The other notable challenge was the emotion model's low confidence on monotone speech. A poetry reading produced 100% low-confidence predictions which is actually correct behaviour. 
The model is trained on conversational speech and struggles with flat, recitative delivery. The confidence gate catches this rather than producing misleading labels.

## Results

Tested on two YouTube sources:

**Source 1 — LibriVox poetry reading (CC-BY)**
- Licence tier: OPEN — passed consent gate
- Duration: 80 seconds
- Diarization: 23 segments, 1 speaker detected
- Emotion tagging: 16 segments tagged, 7 skipped (too short)
- Confidence: 100% of tagged segments fell below the 0.6 threshold
- Outcome: segments flagged as LOW_CONFIDENCE, would be quarantined from training data

**Source 2 — Standard YouTube video**
- Licence tier: UNKNOWN — blocked at consent gate
- No diarization or emotion tagging performed
- Logged in provenance manifest with blocked reason

The low confidence finding on the poetry reading is actually meaningful, it shows the pipeline correctly identifying that monotone, non-conversational speech produces unreliable emotion annotations. 

## Impact

Every training dataset has provenance somewhere. This pipeline makes it explicit and structural. You can't accidentally process a proprietary source because the gate runs before the models do, not after.

The confidence flagging means downstream filtering is easy. Instead of manually reviewing annotations or discovering bad labels after a training run fails, you get a `emotion_flag` field on every segment that tells you exactly why it might be unreliable.

The RTTM output format means diarization results plug directly into standard speech benchmarking tools without any conversion. The JSON annotations are flat and simple enough to load into any training framework without preprocessing.

## Limitations

- The emotion model is English-biased and performs poorly on non-English speech
- All inference runs on CPU — GPU support would reduce processing time significantly on longer files
- The confidence threshold (0.6) is fixed globally; per-emotion adaptive thresholds would be more accurate
- Licence classification relies on yt-dlp metadata, which isn't always populated correctly by uploaders
- The pipeline processes one file at a time; parallel batch processing would be needed for production scale
- Speaker diarization struggles with overlapping speech and very short segments

## Technology and Tools

| Category | Tools |
|---|---|
| Audio acquisition | yt-dlp, FFmpeg |
| Audio processing | soundfile, librosa |
| Speaker diarization | pyannote.audio, pyannote/speaker-diarization-3.1 |
| Emotion recognition | HuggingFace Transformers, wav2vec2-lg-xlsr-en-speech-emotion-recognition |
| ML framework | PyTorch, torchaudio |
| Data outputs | JSON, RTTM, CSV |
| Testing | pytest |
| Language | Python 3.11 |
