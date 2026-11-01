# Handoff: evaluating Argus on existing benchmarks

This branch holds only this file. It describes work to do; no eval has been run. It was written on 2026-10-31 against Argus `main` at f78dfd0, from web research done that day. Delete this file once the sweep lands.

"Verified" below means the fact was read on the source page on 2026-10-31. "Inferred" means it is an estimate or a judgement.

## Goal

Back the claim that Argus is the most advanced robot-data annotation harness with numbers that someone else's labels produce. Writing new human labels is out of scope, because Argus's output is too dense to label by hand. Instead, score the parts of Argus's output that existing human-labelled benchmarks already cover, one benchmark per output type, with one frozen configuration and no tuning for any benchmark.

No single benchmark judges density, so the claim rests on the sweep as a whole. The most credible results come from benchmarks whose labels or scoring someone else holds (Ego4D Goal-Step, OpenGVL). The most distinctive come from label types only Argus produces (mistakes in EgoPER, outcomes in FailBench). WGO-Bench is the familiar headline.

## Protocol (applies to every benchmark)

- Freeze the Argus commit, prompts, frame sampling and model settings before the first scored run. Tune only on a dev split that is separate from every benchmark. For WGO-Bench that has to come from outside the benchmark, for example extra DROID episodes.
- Write only an adapter from Argus's episode JSON to each benchmark's prediction format. No benchmark-specific prompt text.
- Every results table has a harness column and a model column. Show each harness gain against the strongest baseline on the same model: a plain single-call prompt, and Midcentury's pipeline for WGO-Bench.
- Report seeds (at least 3 where an LLM judge is involved), 95% bootstrap confidence intervals, cost per episode, the parse-failure rate and per-family breakdowns.
- Publish the adapters, the scorer (or a reimplementation) and the raw outputs.
- Run after the in-flight Argus work on the primary Mac has merged, since that work changes behaviour, and name the commit in every table.

## Coverage map

| Argus output | Benchmark | Who scores | Best published number |
|---|---|---|---|
| Action timeline and boundaries | WGO-Bench | Self-scored | 50.2% semantic F1 micro (Midcentury's Argus) |
| Steps across long video | Ego4D Goal-Step | Official challenge, held-out test | about 56% R@1, zero-shot |
| Task progress | OpenGVL | Hidden datasets, run by maintainers on request | Leaderboard page has no entries |
| Operator mistakes | EgoPER | Self-scored, public labels | AUC 73.8 / EDA 66.7 (supervised, ICLR 2026) |
| Episode outcome | FailBench | Self-scored, public labels | 0.77 balanced accuracy, best of 13 VLMs |

## WGO-Bench (verified unless marked)

What it is:
- Made by Macrodata Labs. There is no paper and no eval repo. The method is in a blog post (https://macrodata.co/blog/annotating-robot-video-subtasks), and the annotation pipeline (not the scorer) is in the `macrodata-labs/refiner` repo.
- Dataset: https://huggingface.co/datasets/macrodata/WGO-Bench, licensed CC-BY-NC-SA-4.0, so labels derived from it cannot be used commercially.
- 100 episodes, 743 gold segments, 63 instructions, one `train` split, about 1.4GB. The episodes are 50 DROID, 25 Galaxea R1 Lite and 25 HomER egocentric.
- Each row holds the video, the instruction, robot state, and gold segments as start, end and label. Segments mark completed manipulation events.
- There is no hidden test set and no leaderboard. Macrodata tuned its own prompts on a separate 15-episode validation set.

Metric:
- Segment F1: a prediction matches a gold segment at IoU >= 0.75, ignoring labels. The first predicted start and the last predicted end are snapped to the human boundaries.
- Semantic F1: also requires the label to pass an LLM judge (`gemini-3.5-flash`) with a Scale-style rubric: same action and object, synonyms allowed.
- Midcentury reports both micro (pooled over segments) and macro (mean per clip), and says exact equivalence with the official evaluator is unverified.
- Other teams use other thresholds. Manda reports segment F1 0.629 at IoU 0.5 (https://replicate.com/mandarobotics/robot-episode-labeler/readme), which is not comparable.

Published semantic F1 (micro, all self-reported):

| System | Semantic F1 |
|---|---|
| Macrodata seeded relabel | 16.8% |
| Perceptron | 28.0% |
| Stock Argus, Sol 6.1 High (measured by Midcentury) | 29.8% |
| Vidur | 38.6% |
| Midcentury's Argus adaptation, Sol 6.1 High | 50.2% |

Midcentury's own run (https://www.midcentury.xyz/blog/optimizing-argus) went from 29.8% to 50.2% micro and from 46.3% to 68.7% macro. Segment F1 rose from 36.0% to 57.9%, and label accuracy on time-matched segments from 82.7% to 86.6%. Midcentury calls the run "development-exposed" and its table "context rather than a shared leaderboard". Its counts were 244 matches from 894 predictions, which is useful for checking a reimplemented scorer.

What Midcentury changed (the model stayed Sol 6.1 High; there were no ablations, so no gain can be pinned on one change):
- Sampling went from one frame every 1.5s (robot) or 0.5s (ego) to 4 FPS.
- Grids went from 6 columns to 4 columns with cells up to 448px.
- One pass became three: propose events, refine boundaries with 8 FPS windows of plus or minus 1.5s around each endpoint, then relabel each fixed interval from the previous, current and next frames.

Where the gain came from (inferred):
- Mostly timing. Segment F1 rose 22 points while label accuracy rose 4.
- Short actions are the bottleneck. Recall on actions under 2s went from 25.6% to 43.8%.
- HomER is the weak family at 37.3%, against about 74.5% on DROID and Galaxea, and it drags the micro score down.
- `label/policy.py` now has Sol sample at least every 1s on every rig, so the 29.8% stock figure may describe an older build. Rerun stock before comparing.

Levers that would help Argus on every dataset (inferred): denser sampling, an opt-in boundary-refinement pass at a higher frame rate, relabelling each fixed segment with its neighbours as context (Macrodata measured 61.0% against 56.1% without context), and a better model. Levers that only fit the benchmark: matching WGO's "completed event" granularity, tuning event-size priors to its segment count (about 7 per clip), and tuning anything on these 100 clips.

Is 70% reachable (inferred): semantic F1 is roughly segment F1 times label accuracy (57.9 x 0.866 = 50.2). Reaching 70% micro needs segment F1 near 78 at 90% label accuracy, at IoU 0.75, on ambiguous boundaries, and nobody has published human agreement at that threshold. A realistic range with general levers only is 55 to 62% micro. A 70% headline is only defensible on the macro metric, where Midcentury already reports 68.7%, so any claim must name the metric.

Design tension: the WGO gain comes from multi-pass refinement, which Argus deliberately skips for cost. Choose one: add boundary refinement as an opt-in flag (general, and it takes the WGO number), or keep one pass, argue that WGO scores boundary timing rather than density, and lead with EgoPER and FailBench.

Run plan:
1. Freeze the commit and config. Rerun stock Argus to replace the 29.8% figure.
2. Write the adapter from timeline actions to `{start, end, label}`.
3. Get Macrodata's scorer, or reimplement it from their blog and check it against Midcentury's 244 / 894 counts.
4. Run 3 seeds. Report micro and macro, per family, with 95% bootstrap confidence intervals.
5. Run at least two models (Sol 6.1 and one older model) to separate harness from model.

## Ego4D Goal-Step (verified: challenge and numbers; not verified: a 2027 round or an open test server)

- Task: step grounding in long egocentric video. Map Argus timeline steps to the step queries.
- Metric: R@1 at IoU 0.3 (also report 0.5).
- Best published: HiERO-StepG 56.27% zero-shot (https://arxiv.org/abs/2605.31227), OSGNet 58.61% mean R@1 supervised.
- It had a CVPR 2026 challenge. Check whether its test server takes submissions now. If it does, this is the strongest single result available, because the labels are held out and someone else scores them.
- Cost and effort (inferred): video is large, a few hundred dollars of model calls on the validation split; the adapter takes 10 to 16h.

## OpenGVL (verified: page exists with no entries and hidden datasets)

- Task: per-frame task progress (generative value learning, https://generative-value-learning.github.io). Metric: Spearman rank correlation between predicted progress and frame order, on Open X-Embodiment episodes.
- Its datasets are hidden and run by the maintainers on request (https://opengvl-opengvl.static.hf.space/index.html), so a result cannot be tuned against.
- Argus fit: derive progress from subgoal completion and its progress labels. Adapter 6 to 10h, about $50 to $200 of calls (inferred).
- Step zero: ask the maintainers how they take submissions.

## EgoPER (verified: SOTA figures)

- Task: procedural mistake detection and step segmentation on 5 cooking tasks. Map Argus operator mistakes to error segments and its timeline to step segments.
- Metrics: EDA, AUC and IoU. Supervised SOTA is AUC 73.8 / EDA 66.7 (ICLR 2026, https://en.papernotes.org/ICLR2026/multimodal_vlm/procedural_mistake_detection_via_action_effect_modeling/). ZeProM is a zero-shot VLM rival from June 2026 (https://arxiv.org/pdf/2606.21579).
- This is the closest match to what Argus is for. A zero-shot Argus result near the supervised number is a strong story.
- Cost and effort (inferred): under $200, adapter 10 to 14h.

## FailBench (verified: paper and numbers; not found: a harness link)

- arXiv 2609.03611 (Sept 2026): 2,197 robot attempts from 14 public sources, success or failure labels. The best of 13 VLM detectors reaches 0.77 balanced accuracy.
- The abstract says a harness was released, but no link was found. Find it or request it from the authors.
- Argus fit: score its per-episode outcome directly. Adapter 8 to 12h (inferred).

## Other candidates

| Benchmark | Fit | Note |
|---|---|---|
| RoboFAC (78K failure QA, about 8K real) | Detect, locate, explain | Paper only, QA format; RoboFAC beats GPT-4o by 34.1%; KITE (2026, https://arxiv.org/pdf/2604.07034) follows up. Adapter 8 to 12h. https://arxiv.org/pdf/2505.12224 |
| AHA / RoboFail | Failure detection | AHA 2024 numbers (https://arxiv.org/pdf/2410.00371); I-FailSense reports 74% real-world (https://arxiv.org/html/2509.16072v2). Adapter about 6h |
| RoboVQA | Weak | Free-form QA scored by BLEU; Thinker-7B BLEU 63.5 (aggregator, unverified) |
| EgoSchema, OpenEQA | Off-thesis | Multiple-choice QA, saturated by frontier models (inferred) |
| REFLECT, Sentinel, FailGen, VLAC, Assembly101, EPIC-Kitchens, HA-ViD, ECoT | Partial | Not checked |

## Recommended order

1. WGO-Bench clean rerun: frozen config, stock and (if built) the boundary-refinement flag, two models.
2. EgoPER and FailBench: the label types only Argus makes.
3. Ego4D Goal-Step on its test server, if it is open, and OpenGVL through its maintainers: the externally scored results.

If only one can be done, do Goal-Step on a held-out server.
