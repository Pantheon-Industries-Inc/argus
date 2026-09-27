# robot-data-audit

![The board, showing a Galaxea teleop episode with its three cameras, the dense timeline and the outcome against the given goal](media/board.jpg)

Audit robot-learning datasets episode by episode. A vision-language model annotates every episode densely, with a timeline of every action, key events, state changes, the outcome judged against the given instruction, data issues (faults of the recording or its labels) and operator mistakes (a demonstration done badly). Deterministic checks run beside it, on camera streams against the recorded motion, jumps in the recorded state, capture quality and sped-up recordings. A board plays each episode's cameras in sync with its annotation and filters a dataset by problem.

Three rigs are covered, teleoperated arms, handheld grippers that carry cameras, and head cameras worn by a person. The repository has adapters for nine public datasets and for your own data (a LeRobot dataset or a folder of videos), and the model comparison, which runs the same harness with four models on 193 episodes.

## Install

```bash
git clone git@github.com:Pantheon-Industries-Inc/robot-data-audit.git && cd robot-data-audit
uv sync --frozen                        # Python 3.11.15 and the exact package versions in uv.lock
export OPENROUTER_API_KEYS=sk-or-...    # one or more OpenRouter keys, comma-separated
uv run pytest                           # no network, no model call
```

You also need `ffmpeg` 5.1 or newer on your PATH, for the board's clips and the two MCAP datasets. Keys are read only from `OPENROUTER_API_KEYS`, and anything in it that is not an OpenRouter key is ignored. ABC-130k, 10Kh-RealOmin, Egocentric-100K and Gen-HumanEgo ask you to accept their terms on Hugging Face first; for those, also `export HF_TOKEN=hf_...`. Everything the pipeline writes goes under `data/`, which git ignores.

## Quickstart: one episode per rig

Each rig takes three commands, which prepare the episode from its public dataset, run the deterministic checks and label it. The three labels cost about $1 together.

```bash
# teleoperated arms: a Galaxea R1 Lite plugs a device into a power strip (38 s)
uv run python -m prepare galaxea prepare --episodes configs/quickstart/teleop.txt --out data/episodes/galaxea/quickstart
uv run python -m checks data/episodes/galaxea/quickstart
uv run python -m label --dataset galaxea --episodes data/episodes/galaxea/quickstart --kind smoke --cap 2

# handheld gripper: FastUMI, one gripper opens a toilet lid (11.5 s)
uv run python -m prepare fastumi prepare --episodes configs/quickstart/handheld.txt --out data/episodes/fastumi/quickstart
uv run python -m checks data/episodes/fastumi/quickstart
uv run python -m label --dataset fastumi --episodes data/episodes/fastumi/quickstart --kind smoke --cap 2

# head camera: OpenAoE, a phone worn at the head (34 s)
uv run python -m prepare openaoe prepare --episodes configs/quickstart/head_camera.txt --out data/episodes/openaoe/quickstart
uv run python -m checks data/episodes/openaoe/quickstart
uv run python -m label --dataset openaoe --episodes data/episodes/openaoe/quickstart --kind smoke --cap 2
```

Then build a board over the three runs and open it at http://localhost:8896.

```bash
mkdir -p data/boards/quickstart && cp configs/quickstart/board.json data/boards/quickstart/manifest.json
uv run python -m board build data/boards/quickstart
for ds in galaxea fastumi openaoe; do uv run python -m board clips --episodes data/episodes/$ds/quickstart --out data/clips; done
uv run python -m board serve --board data/boards/quickstart --clips data/clips
```

The Galaxea episode should come out as a success then undone (the device is plugged in at about 16 s and unplugged at 28 s, a step the dataset's timed sub-steps describe but its instruction leaves out, so an instruction mismatch). The FastUMI episode should be a clean success with the lid open at 7 s. The OpenAoE clip is usually read as three activities (filling a kettle from a bucket, carrying it to a cabinet, showing the hands), each a success, with the dataset's own action segments matching the footage.

## Layout

| Folder | What it does | Entry point |
|---|---|---|
| `prepare/` | One adapter per dataset: reads exactly the listed episodes and writes each as a sidecar folder | `python -m prepare <adapter>` |
| `checks/` | Deterministic checks, written into each episode before labelling, and the label consistency check | `python -m checks` |
| `label/` | The harness: frame selection, exact decoding, the per-rig prompts, the model call, runs | `python -m label` |
| `board/` | Builds a board from runs, cuts browser clips, serves it or writes it as static files; the hand pose overlay | `python -m board` |
| `compare/` | Other models over the same episodes and harness, with and without an example annotation | `python -m compare` |
| `configs/` | Episode lists, the quickstart, the board manifest template, model settings, example annotations | |
| `tests/` | One test file per stage | `uv run pytest` |

Every command takes `--help`, and every module's docstring gives its inputs, outputs and options.

## Prepare

An adapter reads a dataset and writes one sidecar folder per episode. Nothing is re-encoded. The sidecar points at the dataset's own video files, and the harness decodes each episode's frames out of them by exact timestamp.

### Your own data

```bash
# a LeRobot dataset on disk (v2.0, v2.1 or v3.0)
uv run python -m prepare lerobot prepare --root path/to/dataset --rig teleop_arms --out data/episodes/mine/all
# a folder of head-camera videos, one episode per file, with optional instructions
uv run python -m prepare videos prepare --root path/to/videos --instructions instructions.json --out data/episodes/mine/all
```

`--rig` is `teleop_arms`, `handheld_gripper` or `ego_head`. The LeRobot adapter takes the dataset's video features as cameras, assigned by name (a camera named for a side and a wrist, hand or gripper is that side's mounted camera; the best-named other one is the scene camera), `observation.state` and `action` as the recorded state when they have 7 values per arm or gripper, and each episode's task as its instruction. The instructions file of the videos adapter maps each file's path to its instruction, or to `{"instruction": ..., "subtasks": [{"t0": 0.0, "t1": 4.5, "label": "..."}]}` for timed steps. `--episodes LIST` restricts either to the listed episodes; `--dataset NAME` sets the name the model is told.

For any other format, write the sidecar yourself (the smallest adapters, `prepare/openaoe.py` and `prepare/videos.py`, show how on top of `prepare/sidecar.py`). An episode folder `episode_<name>/` holds:

- `context.json`: `dataset`, `profile` (the rig), `state_kind` (`joints`, `ee_pose` or `none`), `fps`, `n_state_frames`, `cameras` (per view `exo`, `left`, `right`: its `name`, `width`, `height` and `desc`, what the camera is), `task_label`, and the task text if the dataset has one: `instruction`, or `annotation_subtasks` (timed steps). `real_times` names `times.npz` when frames carry real capture times.
- `sources.json`: per view, the video file (`packed`), the episode's offset in it (`base_s`) and its exact frame count (`n_frames`), and a `kmap` file when a camera's frames are paired to the anchor camera by time.
- `state.npz`: `state` and `action`, one row per frame (absent when `state_kind` is `none`).

### The public datasets

```bash
uv run python -m prepare molmo prepare --episodes configs/slices/molmo.txt --out data/episodes/molmo/slice
```

Every adapter has `prepare --episodes LIST --out FOLDER [--raw RAW] [--jobs N] [--force]`, which downloads exactly the episodes in LIST into RAW (default `data/raw/<adapter>`) and prepares them. `configs/slices/<adapter>.txt` is the episode list we audited for each dataset, with the command and seed that drew it in its header. Most adapters also have `sample`, the seeded draw (`python -m prepare <adapter> sample --help`).

| Dataset | Rig | Adapter | `configs/slices` |
|---|---|---|---|
| allenai/MolmoAct2-BimanualYAM-Dataset | teleop | `molmo` | 1,284 episodes, 25.1 h, all 34 tasks |
| XDOF/ABC-130k | teleop | `abc130k` | 183 episodes, 5.9 h, one per task |
| RogersPyke/Galaxea-Open-World-Dataset_10K_20260123 | teleop | `galaxea` | 222 episodes, 5.7 h, 111 collections |
| configinc/HABIT | teleop | `habit` | 315 episodes, 5.1 h, 5 per task |
| IPEC-COMMUNITY/FastUMI_100k_lerobot | handheld | `fastumi` | 964 episodes, 4.8 h, 32 per task |
| genrobot2025/10Kh-RealOmin-OpenData | handheld | `realomin` | 280 episodes, 5.5 h, round robin over task folders |
| builddotai/Egocentric-100K | head camera | `egocentric100k` | 112 clips, 5.6 h, one per worker |
| genrobot2025/Gen-HumanEgo | head camera | `genhumanego` | 79 episodes, 3.5 h, uniform random |
| inclusionAI/OpenAoE-2000h | head camera | `openaoe` | 107 clips, 5.7 h, one per recording |

## Checks

```bash
uv run python -m checks data/episodes/<dataset>/<slice>
```

Each check writes its result into the episode's `context.json`, and `board build` copies it into the board's `dataset_checks`. None of it goes into the prompt.

- `stream_pairing`: does each mounted camera's image motion follow its own arm's or gripper's recorded motion, or the other one's (camera files swapped).
- `recorded_jumps`: single-frame leaps in the recorded state that the actor's own camera does not show.
- `gripper_channels`: a gripper channel with exactly the same value at every frame.
- `capture_qc`: the 38 capture checks of public-dataset-adapter (clock gaps, exposure, frozen or duplicated frames, motion the video does not show), each with its status and a per-rig calibration; the measured reason for every threshold is written next to it in `checks/capture_qc.py`.
- `label_consistency`: annotations that contradict themselves (success beside a goal alignment that says another task was done, success then undone with nothing undone, a failure or partial outcome whose progress reaches 1.0, a time past the end). It reads stored labels, runs inside `board build`, and never edits a label.

MolmoAct2 has one more check. Its recordings can play faster than real time, and the rule compares neighbouring episodes, so it scans the whole dataset (about 32,000 episodes, from the data parquets only):

```bash
uv run python -m checks.timebase scan --raw data/raw/molmo --out data/molmo_timebase.csv
uv run python -m checks.timebase apply --timebase data/molmo_timebase.csv data/episodes/molmo/slice
```

## Label

```bash
uv run python -m label --dataset molmo --episodes data/episodes/molmo/slice --kind full --cap 900
```

`--kind dry` builds every request exactly as it would be sent and costs nothing; each output then holds the full prompt in `prompt_text`. `--kind smoke` is a small paid run to read before a full one. A paid run refuses to start from a checkout with uncommitted changes, so every label names the code that made it, and `--cap` is the most it may spend. Anything after `--` goes to the harness (`python -m label.harness --help`), for example `-- --model anthropic/claude-opus-5.5`.

Each run gets its own folder, `data/runs/<dataset>/<YYYYmmdd-HHMM>_<kind>_<commit>`:

- `run.json`: the commit, slice, exact command, model settings and cap, then the billed cost, episodes done and failed, footage hours and cost per hour.
- `out/<episode>.json`: the labels (`labels`: scene, dense timeline, key events, state changes, outcome with goal and undo times, goal alignment, scene graph, recovery, data issues, operator mistakes, instruction variants, performance review), `parse_ok`, what was sent (`config`: cameras, cell size, the exact instants), `dataset_checks`, the instruction graded against, what served the request, and the billed usage and latency. A reply that does not parse is kept as it came, never retried or repaired.
- `out/failed_<episode>.json`: a reply cut off at the output limit, kept for diagnosis.
- `log.txt`.

`--resume RUN --why "..."` finishes a run killed from outside without labelling any episode twice. `python -m label.reparse RUN` re-reads a run's unparsed replies with the current parser (no model call), keeping each original in `out/reparsed_originals/`.

Per episode the model receives one instant every second (every half second on head cameras) plus the first and last frame, packed four instants to a grid image with one row per camera and each column headed with its exact time. Grid cells are 448 px wide for teleop and 256 px for the other rigs; a request that would pass the provider's image-size cap steps down through 384, 320, 288 px and so on until it fits. The first and last instant follow again at up to 768 px, then the episode's facts: cameras, recorded still spans and recorded motion as claims to check, and the instruction or annotation as a claim to check. The instructions each rig shares are in `label/prompts.py`, pinned by hash in `tests/test_label.py`. Expect roughly $20 to $30 per hour of footage on teleop, $28 to $36 on handheld grippers and $13 to $15 on head cameras.

## Board

A board shows exactly the runs its `manifest.json` names; every build regenerates its episode files from them.

```bash
mkdir -p data/boards/mine && cp configs/quickstart/board.json data/boards/mine/manifest.json   # then edit it
uv run python -m board build data/boards/mine
uv run python -m board clips --episodes data/episodes/mine/all --out data/clips
uv run python -m board serve --board data/boards/mine --clips data/clips
```

A manifest lists one entry per dataset, `{"dataset": NAME, "run": RUN, "episodes": FOLDER, "rules": [...]}`, with paths absolute or relative to the board folder; a run given as `../../runs/<dataset>/latest` is that dataset's newest finished run. `configs/quickstart/board.json` is a complete example. The rules apply definitions after labelling without any model call (an operator mistake another finding already counts is capped, a missing instruction is not a fault on a dataset that ships none, a head-camera wearer pausing between activities is minor); `board/rules.py` has each rig's rules and `board/build.py` documents every rule kind and manifest key. The label consistency check runs on every episode at build time.

A problem counts on the board when it is a data issue at medium or high severity, an operator mistake that changes the outcome at medium or high, or any operator mistake at high; the rest stay visible as minor. Each flagged issue belongs to one problem family (`board/families.json`), which is what the board's filter lists. Each episode can be downloaded as JSON and any filtered list as JSON Lines.

The same page can be written as plain files for a CDN, with no server behind it:

```bash
uv run python -m board static media --board data/boards/mine --clips data/clips
uv run python -m board static site --board data/boards/mine --clips data/clips      # data/boards/mine/static/<build id>
```

`board/publish.sh` uploads a static build to any rclone destination.

### Hand pose on head-camera footage

The board can draw 2D hand keypoints over head-camera episodes (the episode page's "Hand pose" switch). They come from [ACE-Ego-Hand](https://github.com/ggxxii/ACE-Ego-Hand), run on Modal GPUs by `board/hand_pose/modal_app.py`, whose docstring has every command: `python -m board.hand_pose.specs` writes each episode's camera model (the dataset's own fisheye calibration where it has one), and the Modal app fetches the model's public weights, labels the clips and pulls the keypoints to a local folder. Name that folder in the manifest, `"hands": {"src": "../../hand_pose/keypoints", "clips": "../../clips"}`, and `board build` writes `BOARD/hands/` (every keypoint within 0.5 px of the model's, each file timed against the clip the board plays), apart from the labels and never counted or exported. The overlay is display only: the ACE-Ego-Hand weights are CC BY-NC 4.0 and MANO is licensed for non-commercial research, so neither ships here, you fetch the weights (the app does it) and MANO (mano.is.tue.mpg.de) yourself, and the keypoints are not training labels.

## Model comparison

Four models run on the same 193 episodes (about an hour per rig, `configs/compare/main.json`) through the same harness, with the same prompt, images, reasoning effort and output limit (`configs/models.json`). They are Astra (`openai/gpt-6-astra`, the reference and the model `python -m label` uses), Claude Opus 5.5 (`anthropic/claude-opus-5.5`), GPT-6 Sol (`openai/gpt-6-sol`) and DeepSeek v4.1 flash (`deepseek/deepseek-v4.1-flash`). With `--with-example`, the other three also see one complete Astra annotation of a different episode of the same rig (`configs/examples/`), on a seeded third of the episodes (`configs/compare/third.json`).

```bash
uv run python -m compare prepare --selection configs/compare/main.json          # HF_TOKEN needed
uv run python -m compare label --selection configs/compare/main.json --kind full --cap 100
uv run python -m compare label --selection configs/compare/third.json --with-example --kind full --cap 15
uv run python -m compare board --entries data/runs/compare/main.json data/runs/compare/third_ex.json --out data/boards/compare
uv run python -m board build data/boards/compare
uv run python -m compare.metrics data/boards/compare
```

`prepare` prepares and checks exactly the selected episodes of the nine datasets. Each `label` starts one run per model at once, each with its own `--cap`; on these episodes Astra costs about $90, Claude Opus 5.5 $31, GPT-6 Sol $17 and DeepSeek v4.1 flash $3, and the with-example runs $11, $6 and $1. `board` writes a manifest whose board holds the reference model's labels and every run as a comparison. The other models' labels go to `BOARD/compare/`, never into the board's counts or downloads; the board offers them beside its own on each episode and sums them up in a comparison view. `board clips --episodes data/episodes/compare/main --out data/clips` cuts the clips to watch them.

`compare/metrics.py` measures everything from the run folders, and nothing is retried or repaired. It reports the parse share (a reply that does not parse counts against the model), schema violations, timeline events per minute, key events, subgoals, data issues and operator mistakes per episode, outcome and issue-type agreement for every pair of models, cost and latency, and each model's change when given the example. Densities and counts use the episodes every model parsed; agreement uses the episodes both of a pair parsed.

## Determinism

Everything before the model call is deterministic and the same on any machine. The episode lists are fixed files with their seeds, the adapters write the same sidecars from the same dataset files, and the harness builds byte-identical requests, since packages are pinned in `uv.lock`, frames are decoded by exact timestamp, and the font drawn on the grid images ships with the repository (`label/fonts/`). A dry run shows exactly what a paid run would send.

What each request sends is fixed by `configs/models.json` and the harness:

- the OpenRouter model id, `openai/gpt-6-astra` by default, and the four ids above in the comparison;
- `reasoning: {"effort": "medium"}` and `max_completion_tokens: 64000`, for every model;
- `response_format: {"type": "json_object"}`, and a prompt-cache breakpoint at the end of the shared instructions.

No temperature, top_p or seed is sent, and no provider is pinned, so each model samples with its provider's own defaults. A sampling setting would not mean the same thing across the four models, because providers honour temperature and seed unevenly, and a model's hosts differ in which they apply. The comparison instead holds equal everything the harness controls. OpenRouter model ids are not dated snapshots, and the model behind an id can change. Each output records what served it, `provider_name`, `model_served`, `system_fingerprint` and `generation_id`; in the runs made with this harness `model_served` was always the requested id and no provider returned a fingerprint, so the served version cannot be read back beyond the date of the run. In the full comparison run every model was served by one provider except DeepSeek v4.1 flash, which OpenRouter routed to four (Parasail 162, Together 23, Novita 5 and StreamLake 2 of 192 episodes), and to a fifth, Modal, in a later small run.

The model's answer is not bit-reproducible. The same request can come back with a different timeline, and where the footage sits close to a line, a different reading. On the quickstart episodes, each labelled four times with this harness, the Galaxea episode came back every time as success then undone with the goal at 16 s, the undo at 28 s and the same instruction mismatch, in 21 to 26 timeline segments; the FastUMI episode every time as a success with no issues, the goal at 7 s three times and 6 s once, in 7 to 9 segments. The identical OpenAoE request, sent six times, came back as three activities five times and as two once, every activity a success, in 23 to 27 segments. The billed cost varies as well, since a repeated request costs less while the provider still holds the shared instructions in its prompt cache ($0.52 and then $0.19 on the Galaxea episode). Compare labels across runs by their fields (outcome, goal and undo times, issue types), not byte for byte, and read segment counts as approximate.

## Credits

The capture checks in `checks/vendor/public_dataset_adapter_qc.py` are Sambhav Gupta's, from public-dataset-adapter (commit 99d0a9e), vendored with the changes listed in that file's header. The grid font is DejaVu Sans Bold (`label/fonts/LICENSE-DejaVu.txt`). The hand pose model is ACE-Ego-Hand. The datasets belong to their publishers (the Hugging Face ids under Prepare); use each under its own license.

## License

Apache-2.0. See `LICENSE`.
