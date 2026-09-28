# robot-data-audit

Dense annotations and data-quality checks for robot-learning episodes, from Pantheon.

Read the [blog post](https://pantheon.inc/research/we-looked-at-everything), browse every label on the [data board](https://pantheon.inc/data-board), or [label your own data](https://pantheon.inc/data-review).

This is the pipeline behind *We Looked at Everything*, where it labelled 3,546 episodes (66.5 hours) from nine public datasets of teleoperated arms, UMI grippers and human ego video. It takes an episode as it was recorded, from a single video with no instruction to a full dataset with instructions and recorded state, and returns a dense timeline with each action judged advancing, wasteful or idle, progress toward the goal, key events and subgoals, operator mistakes, changes a person made to the scene, and the instruction checked against the footage. Deterministic checks run beside the model for what it should not be trusted with, such as recordings that play faster than real time, camera files swapped between arms, a recorded gripper opening that never changes and poor capture.

The labels come from Astra (`openai/gpt-6-astra`) through a harness that tells it what each kind of rig is, what counts as a mistake on it, and how to check the recording against the pixels. Every prompt is in `label/`. The repository also holds the board that plays each episode with its labels, a comparison of four models on the same harness, and the gate, the regression suite the harness is held to.

![The board, showing a MolmoAct2 episode with its three cameras, the dense timeline and the outcome against the given goal](media/board.jpg)

## Install

```bash
git clone git@github.com:Pantheon-Industries-Inc/robot-data-audit.git && cd robot-data-audit
uv sync --frozen                        # Python 3.11.15 and the exact package versions in uv.lock
export OPENROUTER_API_KEYS=sk-or-...    # one or more OpenRouter keys, comma-separated
uv run pytest                           # no network, no model call
```

You also need `ffmpeg` 5.1 or newer. Anything in `OPENROUTER_API_KEYS` that is not an OpenRouter key is ignored. ABC-130k, 10Kh-RealOmin, Egocentric-100K and Gen-HumanEgo ask you to accept their terms on Hugging Face; for those, also `export HF_TOKEN=hf_...`. Everything the pipeline writes goes under `data/`, which git ignores.

## Quickstart

Each rig takes three commands, which prepare the episode from its public dataset, run the deterministic checks and label it. The three labels cost about $1 together; the MolmoAct2 episode first downloads its three camera packs (about 1.4 GB).

```bash
# teleoperated arms: MolmoAct2, two arms push four blocks into a row, then push it apart (37 s)
uv run python -m prepare molmo prepare --episodes configs/quickstart/teleop.txt --out data/episodes/molmo/quickstart
uv run python -m checks data/episodes/molmo/quickstart
uv run python -m label --dataset molmo --episodes data/episodes/molmo/quickstart --kind smoke --cap 2

# UMI: FastUMI, one gripper opens a toilet lid (11.5 s)
uv run python -m prepare fastumi prepare --episodes configs/quickstart/handheld.txt --out data/episodes/fastumi/quickstart
uv run python -m checks data/episodes/fastumi/quickstart
uv run python -m label --dataset fastumi --episodes data/episodes/fastumi/quickstart --kind smoke --cap 2

# human ego: OpenAoE, a phone worn at the head (34 s)
uv run python -m prepare openaoe prepare --episodes configs/quickstart/head_camera.txt --out data/episodes/openaoe/quickstart
uv run python -m checks data/episodes/openaoe/quickstart
uv run python -m label --dataset openaoe --episodes data/episodes/openaoe/quickstart --kind smoke --cap 2
```

Then build a board over the three runs and open it at http://localhost:8896.

```bash
mkdir -p data/boards/quickstart && cp configs/quickstart/board.json data/boards/quickstart/manifest.json
uv run python -m board build data/boards/quickstart
for ds in molmo fastumi openaoe; do uv run python -m board clips --episodes data/episodes/$ds/quickstart --out data/clips; done
uv run python -m board serve --board data/boards/quickstart --clips data/clips
```

The MolmoAct2 episode should come out as a success then undone, the row complete at about 18 s and pushed apart by about 30 s. The FastUMI episode is a clean success with the lid open at about 7 s. The OpenAoE clip reads as a few activities (filling a kettle from a bucket, carrying it to a cabinet counter, showing empty hands), each a success. A bare video works the same way, with `prepare videos --rig RIG` and no instructions file, and the model names the task itself.

## Layout

| Folder | What it does | Entry point |
|---|---|---|
| `prepare/` | One adapter per dataset and two for your own data, writing episodes as sidecar folders | `python -m prepare <adapter>` |
| `checks/` | Deterministic checks, written into each episode before labelling, and the label consistency check | `python -m checks` |
| `label/` | The harness: frame selection, exact decoding, resolution routing, the per-rig prompts, the model call, runs | `python -m label` |
| `board/` | The board: build, clips, serve or write static files; the hand pose overlay | `python -m board` |
| `compare/` | Other models over the same episodes and harness, with and without in-context learning from an Astra trace | `python -m compare` |
| `gate/` | The regression suite: frame-verified cases and a cost sample per rig | `python -m gate` |
| `configs/` | Episode lists, the quickstart, model settings, example annotations | |

Every command takes `--help`, and every module's docstring documents it. `tests/` has one file per stage.

## Prepare

An adapter reads a dataset and writes one sidecar folder per episode. Nothing is re-encoded: the sidecar points at the dataset's own video files, and the harness decodes each episode's frames out of them by exact timestamp.

```bash
# your own LeRobot dataset (v2.0, v2.1 or v3.0)
uv run python -m prepare lerobot prepare --root path/to/dataset --rig teleop_arms --out data/episodes/mine/all
# your own videos, one episode per file, with optional instructions
uv run python -m prepare videos prepare --root path/to/videos --rig ego_head --instructions instructions.json --out data/episodes/mine/all
# a public dataset, the episode list we audited
uv run python -m prepare molmo prepare --episodes configs/slices/molmo.txt --out data/episodes/molmo/slice
```

`--rig` is `teleop_arms`, `handheld_gripper` or `ego_head`. The LeRobot adapter assigns the video features to cameras by name, reads `observation.state` and `action` when they have 7 values per arm or gripper, and takes each episode's task as its instruction. The videos adapter takes each file as the one camera of its rig; its instructions file maps a file's path to an instruction, or for human ego video to `{"instruction": ..., "subtasks": [{"t0": 0.0, "t1": 4.5, "label": "..."}]}`, and a file with no entry is labelled without one. For any other format, write the sidecar yourself on top of `prepare/sidecar.py`, which documents every field (`prepare/openaoe.py` is the smallest adapter).

Every public adapter has `prepare --episodes LIST --out FOLDER [--raw RAW] [--jobs N] [--force]` and downloads exactly the listed episodes. `configs/slices/<adapter>.txt` is the list we audited, with the command and seed that drew it in its header.

| Dataset | Rig | Adapter | `configs/slices` |
|---|---|---|---|
| allenai/MolmoAct2-BimanualYAM-Dataset | teleop | `molmo` | 1,284 episodes, 25.1 h, all 34 tasks |
| XDOF/ABC-130k | teleop | `abc130k` | 183 episodes, 5.9 h, one per task |
| RogersPyke/Galaxea-Open-World-Dataset_10K_20260123 | teleop | `galaxea` | 222 episodes, 5.7 h, 111 collections |
| configinc/HABIT | teleop | `habit` | 315 episodes, 5.1 h, 5 per task |
| IPEC-COMMUNITY/FastUMI_100k_lerobot | UMI | `fastumi` | 964 episodes, 4.8 h, 32 per task |
| genrobot2025/10Kh-RealOmin-OpenData | UMI | `realomin` | 280 episodes, 5.5 h, round robin over task folders |
| builddotai/Egocentric-100K | human ego | `egocentric100k` | 112 clips, 5.6 h, one per worker |
| genrobot2025/Gen-HumanEgo | human ego | `genhumanego` | 79 episodes, 3.5 h, uniform random |
| inclusionAI/OpenAoE-2000h | human ego | `openaoe` | 107 clips, 5.7 h, one per recording |

## Checks

`uv run python -m checks EPISODES` writes each check's result into the episodes' `context.json`, which `board build` copies to the board; none of it reaches the prompt.

- `stream_pairing`: whether each mounted camera's image motion follows its own arm's or gripper's recorded motion, or the other one's (camera files swapped).
- `recorded_jumps`: single-frame jumps in the recorded pose that the actor's own camera does not show.
- `gripper_channels`: a recorded gripper opening with the same value at every frame.
- `capture_qc`: the 38 capture checks of public-dataset-adapter (clock gaps, exposure, frozen or duplicated frames, motion the video does not show), calibrated per rig with the measured reason beside each threshold in `checks/capture_qc.py`.
- `label_consistency`: labels that contradict themselves (success beside a goal alignment that says another task or only part of it was done, "aligned" beside footage that does not show the goal, success then undone with nothing undone, a failure whose progress reaches 1.0, a time past the end). It reads stored labels inside `board build` and never edits one.

MolmoAct2 recordings can play faster than real time, and that rule compares neighbouring episodes, so it scans the whole dataset from its data parquets: `python -m checks.timebase scan --raw data/raw/molmo --out data/molmo_timebase.csv`, then `python -m checks.timebase apply --timebase data/molmo_timebase.csv EPISODES`.

## Label

```bash
uv run python -m label --dataset molmo --episodes data/episodes/molmo/slice --kind full --cap 900
```

`--kind dry` builds every request as it would be sent and costs nothing; `smoke` is a small paid run to read before a full one. A paid run refuses a checkout with uncommitted changes, so every label names the code that made it, and `--cap` is the most it may spend; anything after `--` goes to the harness, for example `-- --model anthropic/claude-opus-5.5`. A run's folder, `data/runs/<dataset>/<YYYYmmdd-HHMM>_<kind>_<commit>`, holds `run.json` (command, settings, cap, cost per footage hour) and one `out/<episode>.json` per episode with the labels, `parse_ok`, what was sent (cameras, cell size, the exact instants, contact views, the routing answer), the checks, what served the request and the billed usage. A reply that does not parse is kept as it came, never retried or repaired. `--resume RUN --why "..."` finishes a run killed from outside, and `python -m label.reparse RUN` re-reads unparsed replies with the current parser.

What the model receives: one instant every 1.5 s on teleop arms, every second on UMI grippers and every half second on human ego video, plus the first and last frame, four instants to a grid image with one row per camera and each column headed with its exact time. UMI cells are 320 px wide, human ego cells 256 px. A teleop episode's width comes from its task text: a small model (`openai/gpt-6-sol`) reads only that text and says whether the task needs fine detail (lettering or a display, which face of an object is up, small similar objects). Those episodes get 448 px on every camera; the rest get 224 px plus contact views, the scene camera and the acting arm's camera at detail size just after each sharp change of the recorded gripper value. A request over the provider's image-size cap steps down to narrower cells until it fits. The request opens with the shared instructions and the episode's facts (cameras, recorded still spans and motion, the instruction and the objects it names), each as a claim to check; the grids follow, then the first and last instant at up to 768 px with the contact views between them. With no instruction, the model names the task as the most specific end state the demonstrator worked toward. The shared instructions are pinned by hash in `tests/test_label.py`. Measured on first sends of the gate's cost samples, expect about $26 per hour of footage on teleop, $30 on UMI and $19 on human ego video.

## Board

```bash
mkdir -p data/boards/mine && cp configs/quickstart/board.json data/boards/mine/manifest.json   # then edit it
uv run python -m board build data/boards/mine
uv run python -m board clips --episodes data/episodes/mine/all --out data/clips
uv run python -m board serve --board data/boards/mine --clips data/clips
uv run python -m board static media --board data/boards/mine --clips data/clips   # the same page as plain files
uv run python -m board static site --board data/boards/mine --clips data/clips    # data/boards/mine/static/<build id>
```

A board shows the runs its `manifest.json` names, one entry per dataset, `{"dataset": NAME, "run": RUN, "episodes": FOLDER, "rules": [...]}`, paths absolute or relative to the board folder (`../../runs/<dataset>/latest` is the newest finished run). Its rules apply definitions after labelling (`board/rules.py`; `board/build.py` documents every rule and key). A problem counts when it is a data issue at medium or high severity or an operator mistake that changes the outcome at medium or high, or is high; the rest stay visible as minor, and each flagged issue belongs to one problem family (`board/families.json`), which the filter lists. Each episode downloads as JSON and any filtered list as JSON Lines, and a dataset in `board/dataset_sources.json` is credited with its publisher and license on the page and in the download. `board/publish.sh` uploads a static build.

Human ego episodes can also show 2D hand keypoints (the "Hand pose" switch, on by default) and offer them as a download. They come from [ACE-Ego-Hand](https://github.com/ggxxii/ACE-Ego-Hand), run on Modal GPUs by `board/hand_pose/modal_app.py`, whose docstring has every command; you register for and download MANO yourself. Name the keypoint folder in the manifest, `"hands": {"src": "../../hand_pose/keypoints", "clips": "../../clips"}`, and `board build` writes the overlay and the downloads, apart from the labels. The keypoints are for non-commercial use only, which every file says (see Licenses).

## Model comparison

Four models run on the same 193 episodes (about an hour per rig, `configs/compare/main.json`) through the same harness, with the same prompt, images, reasoning effort and output limit (`configs/models.json`): Astra (`openai/gpt-6-astra`, the reference), Claude Opus 5.5 (`anthropic/claude-opus-5.5`), GPT-6 Sol (`openai/gpt-6-sol`) and DeepSeek v4.1 flash (`deepseek/deepseek-v4.1-flash`). With `--with-example`, the other three learn in context: each prompt also holds one Astra trace, the complete Astra annotation of a different episode of the same rig (`configs/examples/`), on a seeded third of the episodes (`configs/compare/third.json`).

```bash
uv run python -m compare prepare --selection configs/compare/main.json          # HF_TOKEN needed
uv run python -m compare label --selection configs/compare/main.json --kind full --cap 100
uv run python -m compare label --selection configs/compare/third.json --with-example --kind full --cap 15
uv run python -m compare board --entries data/runs/compare/main.json data/runs/compare/third_ex.json --out data/boards/compare
uv run python -m board build data/boards/compare
uv run python -m compare.metrics data/boards/compare
```

`prepare` prepares and checks exactly the selected episodes, and each `label` starts one run per model at once, each within its own `--cap`; Astra costs about the figures under Label for these three hours, and in our run Claude Opus 5.5, GPT-6 Sol and DeepSeek v4.1 flash cost 34%, 19% and 4% as much per episode. `board` writes a manifest whose labels are the reference model's, with every other run as a comparison under `BOARD/compare/`, never counted or exported; the page's "Labels by" control switches the board to one model's labels and opens the comparison view. `compare/metrics.py` measures, from the run folders alone, parse share, schema violations, density, agreement between models, cost, latency and what in-context learning changes.

## Gate

```bash
uv run python -m gate prepare                        # HF_TOKEN needed
uv run python -m gate label --kind full --cap 40     # one run per rig, about $60 in all
uv run python -m gate score data/runs/gate_teleop/RUN data/runs/gate_handheld/RUN data/runs/gate_ego/RUN
```

The gate holds 126 episodes (`gate/selection.json`) and what each case's label must say (`gate/cases.json`), every fact checked on the frames. Teleop has MolmoAct2 1346 six times (asked to flip three blocks, none ever turns: a failure with a high instruction mismatch), 1213 and 1233 (a goal reached, then undone), MolmoAct2 008276 three times and 010414 (black polo shirts under an instruction to fold black pants: the shirt must be named, and the task asked for was not done) and ten more, among them Galaxea Steam_Rice 000050 and 000011. UMI has all 32 FastUMI Prepare_tableware episodes (a fork is handled, never chopsticks) and four more. Human ego has four clips with a known data issue and three clean ones whose objects must be named. Each rig also has a cost sample of about 20 real episodes across its datasets, priced as first sends. A harness passes when every reply parses, no label contradicts itself, the cases hold and the cost stays at the figures under Label; this one scores 19 of 22 on teleop (1346 is right in three of six samples), 36 of 36 on UMI and 7 of 7 on human ego.

## Determinism

Everything before the model calls is deterministic and the same on any machine: the episode lists are fixed files with their seeds, the adapters write the same sidecars from the same files, and the harness builds byte-identical requests (packages pinned in `uv.lock`, frames decoded by exact timestamp, the grid font shipped in `label/fonts/`). A dry run shows what a paid run sends, with teleop at the wide cells, since it makes no routing call. An episode makes at most two calls, both to OpenRouter with `response_format: {"type": "json_object"}` and the billed usage requested:

- the routing call, teleop only: `openai/gpt-6-sol`, `reasoning: {"effort": "low"}`, `max_completion_tokens: 2000`, one text part with the routing question and the episode's task text (dataset, robot, instruction, task label, timed sub-steps), never a frame. Episodes with the same text share one answer within a run; no task text, a failed call or `--cell-w` means the wide cells.
- the labelling call: the model id (`openai/gpt-6-astra`, or the comparison's), `reasoning: {"effort": "medium"}`, `max_completion_tokens: 64000`, and a prompt-cache breakpoint closing the shared instructions.

No temperature, top_p or seed is sent and no provider is pinned, since providers honour them unevenly. OpenRouter ids are not dated snapshots, so each output records `provider_name`, `model_served`, `system_fingerprint`, `generation_id` and the routing answer with its reason.

The answers are not bit-reproducible. Routed five times each, 30 of 34 teleop task texts got the same answer every time, and every task that turns on which face of a block is up was fine detail all five times. The three quickstart episodes and the three bare clips, each labelled three times, got the same outcome every time, with goal and undo times within 5 s of each other (the bare teleop row at 13.5 or 18 s); the OpenAoE clip came back as three or four activities. A repeated request also costs less while the provider still caches it (the bare human ego clip $0.84, then $0.54). Compare labels across runs by their fields (outcome, goal and undo times, issue types), not byte for byte.

## Credits

The capture checks in `checks/vendor/public_dataset_adapter_qc.py` are Sambhav Gupta's, from public-dataset-adapter (commit 99d0a9e), with the changes listed in that file's header. The hand pose model is ACE-Ego-Hand (Yufei Liu et al., [arXiv:2608.20308](https://arxiv.org/abs/2608.20308)), built on Wan2.2, VideoX-Fun and MANO (Romero, Tzionas and Black, 2017).

## Licenses

The code in this repository is Apache-2.0 (`LICENSE`). A few files contain material under its own license, listed in `THIRD_PARTY_NOTICES.txt`. Everything below is downloaded when you run the pipeline and is not included here.

The labels Pantheon publishes (on the data board, in its downloads and in the release) are licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), so anyone may use them for any purpose with credit to Pantheon. The footage they describe keeps its dataset's license.

| Component | License | Source |
|---|---|---|
| ACE-Ego-Hand code (commit 9757868) | MIT | [github.com/ggxxii/ACE-Ego-Hand](https://github.com/ggxxii/ACE-Ego-Hand/blob/97578680931b3f1c8111396c100d595b98857fb8/LICENSE) |
| ACE-Ego-Hand checkpoints (`ace_ego_hand_k.pt`, `ace_ego_hand_kfree.pt`) | CC BY-NC 4.0, non-commercial | [huggingface.co/acerobotics2025/ACE-Ego-Hand](https://huggingface.co/acerobotics2025/ACE-Ego-Hand) |
| Wan2.2-Fun-5B-Control (weights, VAE, umT5-XXL text encoder and tokenizer) | Apache-2.0 | [huggingface.co/alibaba-pai/Wan2.2-Fun-5B-Control](https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-Control) |
| VideoX-Fun (commit 968f0e2) | Apache-2.0 | [github.com/aigc-apps/VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) |
| smplx 0.1.28 | Max Planck license for non-commercial scientific research | [github.com/vchoutas/smplx](https://github.com/vchoutas/smplx/blob/main/LICENSE) |
| MANO (you register and download it yourself) | Max Planck license for non-commercial scientific research, no redistribution | [mano.is.tue.mpg.de/license.html](https://mano.is.tue.mpg.de/license.html) |
| Python packages in `uv.lock` | BSD, MIT, Apache-2.0, PSF, MPL-2.0 (certifi, tqdm) | PyPI |
| MolmoAct2-BimanualYAM-Dataset (Ai2) | Apache-2.0 | [allenai/MolmoAct2-BimanualYAM-Dataset](https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset) |
| ABC-130k (XDOF) | Apache-2.0 | [XDOF/ABC-130k](https://huggingface.co/datasets/XDOF/ABC-130k) |
| Galaxea Open-World Dataset (Galaxea) | CC BY-NC-SA 4.0, non-commercial | [OpenGalaxea/Galaxea-Open-World-Dataset](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset); the adapter reads the copy at `RogersPyke/Galaxea-Open-World-Dataset_10K_20260123`; accept Galaxea's terms on its page before using it |
| HABIT (Config) | CC BY 4.0 | [configinc/HABIT](https://huggingface.co/datasets/configinc/HABIT) |
| FastUMI-100K | Apache-2.0 | [IPEC-COMMUNITY/FastUMI_100k_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/FastUMI_100k_lerobot) |
| 10Kh-RealOmin-OpenData (GenRobot) | CC BY-SA 4.0 | [genrobot2025/10Kh-RealOmin-OpenData](https://huggingface.co/datasets/genrobot2025/10Kh-RealOmin-OpenData) |
| Egocentric-100K (Build AI) | Apache-2.0 | [builddotai/Egocentric-100K](https://huggingface.co/datasets/builddotai/Egocentric-100K) |
| Gen-HumanEgo (GenRobot) | CC BY-SA 4.0 | [genrobot2025/Gen-HumanEgo](https://huggingface.co/datasets/genrobot2025/Gen-HumanEgo) |
| OpenAoE-2000h (inclusionAI) | Open-AoE Dataset License, attribution required; its MANO-derived files, which the adapter never reads, also fall under MANO's license | [inclusionAI/OpenAoE-2000h](https://huggingface.co/datasets/inclusionAI/OpenAoE-2000h/blob/main/LICENSE) |

Running a dataset through this pipeline does not change its license, so credit each dataset as its license asks when you publish its footage or labels. The hand keypoints are predictions of the ACE-Ego-Hand checkpoints, which are licensed CC BY-NC 4.0, from a model that uses MANO, which is licensed for non-commercial research only, so use them only for non-commercial purposes and credit ACE-Ego-Hand.
