<h1 align="center">Argus</h1>

<p align="center">
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.11-455468" alt="Python 3.11"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-455468" alt="Apache-2.0"></a>
  <br>
  <a href="https://pantheon.inc/research/argus"><img src="https://img.shields.io/badge/Blog-0969da" width="46" alt="Blog"></a>
  <a href="https://pantheon.inc/data-board"><img src="https://img.shields.io/badge/Dashboard-1a7f37" width="90" alt="Dashboard"></a>
  <a href="https://huggingface.co/datasets/Pantheon-inc/DiverseUMI-sample100"><img src="https://img.shields.io/badge/DiverseUMI--sample100-8250df" width="181" alt="DiverseUMI-sample100"></a>
</p>

> [!IMPORTANT]
> Argus has many tunable command-line options. Please read [Settings](#settings) in full.

## Updates

<details><summary><b>We release DiverseUMI-sample100, 99.6 hours of bimanual UMI labeled by Argus</b> (October 2026)</summary>

We released [DiverseUMI-sample100](https://huggingface.co/datasets/Pantheon-inc/DiverseUMI-sample100), 99.6 hours of bimanual UMI annotated by Argus at Sol 6.1 High. It spans 12,219 episodes across 179 skills, each with dense action, outcome and progress labels, key events, object state changes, scene graphs and operator recoveries, alongside SLAM-tracked wrist poses and gripper actions.

</details>

<details><summary><b>TU Munich releases Argus labels for 5,000 hours of DROID, ABC-130k and other robot data</b> (October 2026)</summary>

Kalyan V. Nadimpalli (TU Munich) released [Argus annotations](https://huggingface.co/datasets/nkalyanv/argus-robot-annotations) (Sol 6.1 Medium) for more than 5,000 hours of public robot data, which covers 290k episodes across ABC-130k, MolmoAct2 YAM, DROID, GM-100 and ABC-Sim. The labels are released under CC-BY-4.0.

</details>

<details><summary><b>An adaptation of Argus sets SOTA on WGO-Bench at 50.2% semantic F1</b> (October 2026)</summary>

Recently, [Midcentury AI](https://www.midcentury.xyz/blog/optimizing-argus) pushed an adaptation of Argus to SOTA (**50.2% semantic F1**) on WGO-Bench. Because this result uses **Sol 6.1 High**, its advantage over older-model baselines cannot be attributed to the harness alone.

</details>

<details><summary><b>Argus now labels depth, tactile and other modalities and grades how well each episode was executed</b> (October 2026)</summary>

Argus now annotates depth, tactile and other sensor data jointly with video, state and actions. Before labelling, the model can make an inspection call that selects recorded fields to read at full rate. By default Argus makes this call only for a recording with a touch sensor or a field the prompt omits, such as EgoDex hand poses.

Depth is not inspected by default, because its pictures already go into the request, and inspecting it on the Unitree Go2 datasets cost about $0.11 an episode on top of a $0.15 label. The prompt carries force/torque and other numeric channels as sampled readings and the times of their extremes. Choose [`--inspection all`](#settings) when a finding depends on exact depth, force or joint values. Before it spends, review prints how many requests carry an inspection and why, and run.json keeps the same counts.

Argus also scores each episode out of 10 for execution and outcome, and each subgoal for execution, deterministically from the annotations. Outcome measures whether the goal was reached and how far progress got, and execution weights cleanliness 75%, steadiness 15% and speed 10% ([`board/scores.py`](board/scores.py) calls them cleanliness, fluency and efficiency). A subgoal is a lasting state change the goal depends on, such as a block placed in the row, never a pick or a grasp.

</details>

## Overview

To my knowledge, Argus is the most advanced open-source robot learning annotation harness (as of 2026).

It is the only harness that combines native multi-format ingestion ([LeRobot's annotation pipeline](https://huggingface.co/docs/lerobot/main/annotation_pipeline), for example, accepts only LeRobot), joint visual/sensor reasoning, dense annotations of **actions, subgoals, outcomes, operator mistakes, recoveries and object-state changes**, and execution and outcome scores for each episode, with execution scores for each subgoal. It reads multi-view RGB alongside state/action trajectories, depth, tactile and other recorded fields, and keeps their timing and measurement semantics. For tactile data, and for depth with `--inspection all`, the VLM can inspect additional samples or image regions to support a finding, and the viewer and JSON exports keep each finding's evidence, confidence and limitations.

Each request covers one recording, or one part of a long one. Argus samples a frame every 1.5s on teleop, 1s on UMI and 0.5s on human ego, plus the edges of still spans and the first and last frames, and tiles them into JPEG grids with one row per camera and one timestamped column per instant, four instants to a grid (six with one camera). Cells are 224px wide on teleop, 320px on UMI and 256px on human ego. The first and last instants follow the grids at larger size, and the text adds the instruction as a claim to check, the recorded motion between instants and the sensor readings. A fixed prompt comes first, with rules for what counts as a data issue on each rig and variants for an instruction, timed subtasks and recorded state; human ego has its own output schema.

<p align="center">
  <a href="https://raw.githubusercontent.com/Pantheon-Industries-Inc/argus/refs/heads/main/media/architecture.svg?v=a088e8f800e6">
    <img src="media/architecture.svg?v=a088e8f800e6" width="1900" alt="Annotation pipeline: format ingestion, field interpretation (off by default), visual, state and sensor evidence, evidence inspection (automatic for touch recordings and for fields the prompt does not carry), VLM annotation of behavior, state changes and sensor findings with a separate regional pass for tactile arrays, issue detection, temporal stitching of request parts and of the tasks that continue across them, and a viewer with chosen sensor cards, quality scores and JSON exports.">
  </a>
</p>

| Input support | Details |
|---|---|
| Recording types | Teleop, UMI and human ego |
| Native formats | LeRobot v2.0/v2.1/v3.0, MCAP, ROS 2 bags stored as SQLite (.db3, read by the MCAP reader), HDF5, video files and folders |
| Archives | ZIP and TAR, plain or compressed with gzip, bzip2 or xz |
| Sidecar metadata | JSON, JSON Lines, CSV, TSV, text and Markdown notes, and LeRobot's Parquet and JSON Lines annotation tables |
| Public dataset adapters | Ten, each downloading and preparing a public dataset |

| Available evidence | How it contributes |
|---|---|
| RGB | Timestamped multi-view frames provide visual context for task execution, object interactions, progress, failures and recoveries. |
| Tactile, inspected by default | Pressure, normal/shear force, contact arrays and optical tactile images support analysis of grip, contact distribution and surface motion. The model compares whole arrays, single cells and regions with video and gripper state, and reads optical tactile images at the moments they change most. Array shape alone does not establish anatomy or force, and response changes alone do not establish slip. |
| Depth, inspected with `--inspection all` | Metric depth supports camera-relative distance and separation when units, calibration and temporal alignment are known. Relative depth remains relative. |
| Proprioception and actions | Joint states, end-effector poses, gripper states and action trajectories provide motion context. Training targets keep their declared semantics and are not assumed to be executed commands. |
| Other episode fields | The prompt gives each force/torque, numeric, event flag, waveform or structured metadata channel as its range over the recording, plus a reading at each sampled instant when the sampling rate resolves the channel, and otherwise its window means or extremes. [`--inspection all`](label/policy.py) also inspects a recording whose only extra fields are these. |
| Task context | The prompt presents instructions, annotations, calibration and the uploader's own metadata fields as claims, and the VLM checks each one against the recording. The statistics, storage offsets and dataset totals LeRobot writes into its metadata stay in the episode's context.json and out of the prompt. |

## Install and review

You need Python 3.11, [uv](https://docs.astral.sh/uv/), and [`ffmpeg`](https://ffmpeg.org/ffmpeg.html)/[`ffprobe`](https://ffmpeg.org/ffprobe.html) 6.1 or newer. Board clips keep each frame's source timestamp with `-enc_time_base demux`, which older ffmpeg refuses, so `python -m review`, `board clips` and `board static media` stop at the start with a message that names the version they found.

```console
git clone https://github.com/Pantheon-Industries-Inc/argus.git
cd argus
uv sync --frozen
```

Paid annotation requires [`OPENROUTER_API_KEYS`](label/harness.py) (comma-separated keys) or [`OPENAI_API_KEY`](label/harness.py) in your shell or secret manager. OpenRouter requests carry an app attribution header ([`OPENROUTER_APP_HEADERS`](label/harness.py)). Gated Hugging Face datasets require accepting their terms and setting [`HF_TOKEN`](prepare/hub.py).

Annotate a dataset under a spending cap, then open its board. Add [`--free`](review/__main__.py) to build the base requests without model calls.

```console
uv run python -m review --data path/to/dataset --rig teleop_arms \
  --dataset-note "Two arms arrange blocks on a tray." --out data/review/mine --cap 20
uv run python -m board serve --board data/review/mine --clips data/review/mine/clips
```

[`--data`](review/__main__.py) accepts a folder, file, archive or HTTP(S) URL. Set [`--rig`](review/__main__.py) to [`teleop_arms`](review/__main__.py), [`handheld_gripper`](review/__main__.py) or [`ego_head`](review/__main__.py) for teleop, UMI or human ego, respectively. Argus detects the format and every stream a recording carries, and works from the video alone when a recording has nothing else.

**We strongly recommend defisheyeing fisheye footage before labelling it.** The model sees each sampled frame downscaled to one cell of a frame grid, a few hundred pixels wide, and a raw fisheye frame spends much of that cell on its black border and compressed edges.

### Settings

| Setting | Default | Options |
|---|---|---|
| [`--field-dictionary`](review/__main__.py) | `off` | <img src="https://img.shields.io/badge/most_important_setting-b42318" alt="Most important setting"><br>**Turn on for recordings with unusual sensors.** `on` sends one text request per upload (at most $3, no frames) that asks a model what each recorded field beyond the cameras means, such as which signal is a touch sensor. Turn it on for tactile, audio, force or other custom streams whose names do not say what they are. Ordinary teleop, UMI and human ego recordings label well with it off, and fields are then labelled as recorded |
| [`--model`](label/policy.py) | `openai/gpt-6-astra` | `openai/gpt-6.1-sol` (Sol 6.1), or `google/gemini-4-argon` (Gemini 4 Argon) with an OpenRouter key, priced from OpenRouter's published rates ([`label/pricing.py`](label/pricing.py)) |
| [`--reasoning`](label/policy.py) | Medium for Astra, high for every other model | `none`, `minimal`, `low`, `medium`, `high` or `xhigh` |
| [`--sample-fps`](label/policy.py) | The rig's period in the [Overview](#overview); Sol 6.1 samples every 1s on every rig, human ego included, and refuses sparser sampling | `2` catches subsecond fumbles and rapid recoveries, at the cost of more images and, on teleop and UMI, shorter request windows |
| [`--inspection`](label/policy.py) | `auto`, for recordings with a touch sensor or with fields the prompt does not carry | `all` also inspects depth, force/torque, arm signals, actions and source metadata, for runs whose findings need exact values; `off` makes no evidence selection call, though a tactile array still gets its separate regional pass |
| [`--resolution-route`](label/harness.py) | `off`, every teleop episode at the narrow cells (224px) with contact detail views and no routing call | `on` asks a small model, from the episode's task text alone, whether judging it needs fine detail such as lettering or which side of an object is up, and sends those episodes at 448px |
| [`--service-tier`](review/__main__.py) | Unset, the provider's default tier | `flex` sends the same request at half the price with lower priority, and the provider can reject it when it has no capacity. It is independent of `--model` and `--reasoning`, the cap reserves flex rates, and each label records the tier that served it, with counts in `.service_tier/counts.json` in the labels folder |
| [`--flex-fallback`](review/__main__.py) | `off`, and after about 4 minutes of retries a rejected unit stays unlabelled and unpaid, and a resume sends it again | `standard` sends it once at the standard tier, at twice the price, and the cap reserves that rate for every request. A flex request that timed out is never sent again at standard, since it may still be billed |
| [`--allow-unverified-fees`](review/__main__.py) | Off, and no request goes to a model whose published search fee could apply, since no cap can limit that fee | On sends those requests with a printed warning. Normal runs do not need it, because a request that sends no search option cannot be charged for one, which covers OpenAI, Claude and Gemini models. On or off, a run through OpenRouter stops at the first reply whose usage counts a search |
| [`--cap`](review/__main__.py) | $20 | The most labelling may spend, counting every call and outstanding reservation |
| [`--max-minutes`](review/__main__.py) | All footage | Reads at most this many minutes of footage |
| [`--dataset`](review/__main__.py) | The name of `--data` | The name shown on the board and told to the model |
| [`--dataset-note`](label/policy.py) | The source's own collection note, if any | A description of the recordings given to the model as context |
| [`--max-tokens`](review/__main__.py) | 64,000 | The output limit of each labelling request; a smaller limit reserves less against the cap |
| [`--timeout`](review/__main__.py) | 6060s | Idle limit for a labelling request's connection, at least 6060s (Cloudflare's 6000s idle limit plus a minute) |
| [`--concurrency`](review/__main__.py) | All of them | The most labelling requests in flight at once. Argus sends every request at once ([`label/inflight.py`](label/inflight.py)), except that the first episode of each prompt variant goes alone, so the others find the shared instructions in the provider's prompt cache ([`cache_warmup`](label/harness.py)). A request holds its encoded frames until the reply arrives, at most about 70MB, so set a number only on a machine short of memory. |
| [`--jobs`](review/__main__.py) | 2 | Decode processes per preparation step |
| [`--grouping`](review/__main__.py) | Inferred from file names | JSON such as `{"cam_folder": "cameras"}` or `{"take_folder": "takes"}`, for folders whose file names cannot tell separate takes from cameras of one take |
| [`--free`](review/__main__.py) | Off | Builds the base requests and stops before any model call |
| [`--board-clips`](review/__main__.py) | `on` | `off` decodes each camera and writes no viewing copy |
| [`--force`](review/__main__.py) | Off | Relabels every selected episode with new paid calls, keeping earlier replies and charges |
| [`--retry-received EPISODE`](review/__main__.py) | None | Sends that episode's request again after an unusable reply the model wrote, or after a provider failure that has used its two automatic resends; repeat for each episode |

Rerunning a review with the same [`--out`](review/__main__.py) folder resumes it and reuses every output whose inputs are unchanged, matched by content hash ([`label/content_identity.py`](label/content_identity.py)). [`--free`](review/__main__.py) previews only the base requests, without field interpretation, routing, inspection or sensor cards.

> [!NOTE]
> `python -m review` does all post-processing itself once the labelling requests return, and nothing else needs to be run. It stitches long recordings, merges data issues that several parts reported, and builds the review board. Before publishing a result, it removes the phrases the model writes about the labelling itself, such as "in this supplied window" ([`label/wording.py`](label/wording.py)), and drops a sentence only when every word in it is about the labelling.

## Labels

Below is part of the label of MolmoAct2 episode 017777 ([`configs/examples/example_teleop_arms.json`](configs/examples/example_teleop_arms.json)), two of its 62 timeline steps and two of its 7 key events. The instruction asks for the scanned items to end in the basket, and the operator leaves them on the table. A subgoal is a key event of kind `subgoal_complete`.

```json
{
  "completion": {
    "task_completed": "failure",
    "success_predicate": "Each of the four packaged items has been picked up and scanned, and all four scanned items are in the basket.",
    "reason": "The four items are scanned, but the demonstration deliberately empties the basket and leaves the scanned items on the right-hand tabletop instead."
  },
  "timeline": [
    {"start_s": 31.5, "end_s": 34.5, "arm": "right", "action": "rotate and present packet to scanner", "object": "green Kleenex tissue packet", "destination": "barcode scanner", "spatial_relation": "over scanning window", "contribution": "advancing", "progress": 0.3},
    {"start_s": 35.5, "end_s": 36.5, "arm": "right", "action": "set down and release", "object": "green Kleenex tissue packet", "destination": "far-right table", "spatial_relation": "beside orange packet", "contribution": "advancing", "progress": 0.3, "notes": "Release inferred between 36 and 37.5 s; scanner window glows cyan."}
  ],
  "key_events": [
    {"t_s": 22.5, "label": "All four items staged outside the now-empty basket", "kind": "subgoal_complete", "outcome": "success"},
    {"t_s": 72.0, "label": "All scanned items left on the right-hand table, not in the basket", "kind": "terminal_state", "outcome": "success"}
  ]
}
```

### Task verdicts

Argus annotates every recording as the list of tasks it observes, and judges each task against what the dataset claims about it. Instructions and timed subtasks come from the dataset's task fields, and a LeRobot text column counts only when its name marks it as the task, such as `language_instruction`. A task text that spans the whole recording is never a subtask, and becomes the recording's instruction when it has none.

A teleop or UMI recording with an instruction is judged as a whole against it, and work the instruction does not ask for never raises that verdict. With timed subtasks, each task is judged against the subtask it covers. With neither, a recording of one task takes that task's outcome, and a recording of several reports each task separately.

A human ego clip of one task takes that task's outcome unless the dataset provides timed subtasks, and a clip of several tasks gets no overall verdict. When the annotation gives a goal, the verdict is whether the clip fully reached it. Without a goal, a task cut off by the end of the clip is named after the work the clip shows.

### Long recordings

Argus cuts a recording too long for one request (450s at default sampling) at still moments, labels each part and stitches the parts back onto the recording's timeline. Neighbouring parts overlap by 15s, and the stitch joins a task or step to its counterpart from the other side when the two share at least half of the shorter one's seconds in the overlap, however differently the requests worded them. Labels made without the overlap have their tasks joined by a small seam check ([`python -m review seam`](label/seam.py)).

A task still under way at a cut that cannot be joined gets the outcome `unclear`, and an instructed teleop or UMI recording labelled in parts gets an unclear overall verdict, since no part sees both its start and its end. A small text-only request merges data issues that several parts report ([`label/issue_groups.py`](label/issue_groups.py)). It spends from what `--cap` leaves after labelling, within its own $5 limit, and when that runs out the issues stay apart.

### Data issue severity

Some severities are fixed minimums, independent of training impact. On [`handheld_gripper`](label/prompts.py), the prompt rates the operator using their own phone, a gripper set down or taken off while the work goes on, and egregious bare-hand or another person's work at least medium. It rates any face in any camera medium under `face_visible`, with one timed issue per stretch.

On every rig, the parser ([`idle_issues_low`](label/idle.py)) rates an idle stretch low and keeps the model's own rating in `severity_as_labelled`. An idle item that describes other activity, such as phone use, another person handling the objects, bare-hand work or a gripper set down, keeps the model's severity, as does a frozen or black camera.

## Results and further use

> [!IMPORTANT]
> The board is the canonical dashboard for Argus annotations, QC results and robot-data episodes. Serve the data with [`board serve`](board/serve.py) or [`board static site`](board/static.py), and extend the [viewer](board/index.html) where a view is missing rather than writing a new page.

Live and static boards share one [viewer](board/index.html), which overlays behavioral and sensor annotations on the footage with the supporting measurements beside it. The caption over the video names the step under way, with one line per side when the arms, grippers or hands do different things at once ([`label/prompts.py`](label/prompts.py)). Each step holds one action, and a hand-over spans several steps.

The banner over the dense timeline gives the model's reading of the episode under the recording's instruction. On an inference rollout it also shows the policy's instruction, whether the rollout succeeded, and how long the policy held the robot still while it computed its action chunks. Below the footage, an episode page lists the outcome and scores, the problems, one table of subgoals and key events, the sensors and scene, then every check that ran. The same findings export as episode JSON, filtered JSON Lines and static sites.

For a run in progress, give a dataset's manifest entry a [`target`](board/build.py) such as `{"episodes": 2682, "hours": 101.9}`, and the board shows the episodes and hours labelled so far against it, even before its first episode is labelled. A page open on `board serve` checks for a rebuild every 30s and updates its list and totals in place, keeping the open episode.

> [!TIP]
> To count the tasks in your labelled data and the hours of each, run [`python -m label.catalog`](label/catalog.py) `--label L --out OUT`. It places every labelled task in a task family, a coarse task and a specific task. A specific task is one action on one kind of object, so picking up a rubber duck and picking up a stuffed animal are counted separately. Each row lists its hours, the hours in which the task succeeded, and how many recordings it appears in. When you label more data, [`--base`](label/catalog.py) adds it to your last catalog under the same task names.

Public exports (the static site, downloads and [`/api/export`](board/serve.py)) name the uploader's own metadata (file attributes, sidecar fields, publisher fields) but leave out its values, because no filter can recognize every secret a person writes there. Add [`--publish-uploader-values`](board/serve.py) to `board serve` or `board static site` to publish them. In every field, exports also redact access attributes, recognizable credential values and private access locations.

Field definitions from the dataset publisher or user override inferred meanings and layouts without changing recorded measurements. Changes to source data or field interpretations invalidate stale findings.

Labels of a LeRobot dataset can go back into that dataset for training. [`python -m board export-lerobot`](board/lerobot_export.py) `--dataset DS --labels L --out OUT` copies the dataset to a new folder and writes the board's episode files into LeRobot's two language columns. The timeline becomes `subtask` rows in `language_persistent`, one active on every frame. Outcomes, operator mistakes, recoveries, key events and object state changes become `language_events` rows in five new styles of the same names (`outcome`, `mistake`, `recovery`, `key_event`, `state_change`), each holding the finding as JSON.

Every time is snapped to a frame timestamp read from the dataset's own data files, and a time more than one frame period outside the episode is refused. The report lists each refused time, each time written more than one frame period from its frame, each step that lost its only frame, and each repeated timestamp. The copy keeps each episode in its own row group, drops the older `subtask_index` and `tools` columns as LeRobot's own annotation writer does, and is checked against the rules of LeRobot's annotation validator.

A project that loads these columns registers the five styles in `EXTENDED_STYLES` and `EVENT_ONLY_STYLES` in `lerobot.datasets.language`. The exporter works only for recordings that started as LeRobot datasets.

![Two-camera UMI annotation walkthrough, with action phases, subgoals and goal reasoning over the footage.](media/board_demo.webp)

[Browse example annotations](https://pantheon.inc/data-board).

<details>
<summary>Individual stages, dataset adapters and custom boards</summary>

**Staged commands**

```console
uv run python -m prepare folder prepare --root path/to/data --rig teleop_arms --out data/episodes/mine/all
uv run python -m checks data/episodes/mine/all
uv run python -m label --dataset mine --episodes data/episodes/mine/all --kind dry
uv run python -m label --dataset mine --episodes data/episodes/mine/all --kind full --cap 20
```

The staged commands do not interpret fields. [`label`](label/run.py) cuts and stitches long recordings as `review` does. [`prepare lerobot`](prepare/__main__.py) and [`prepare videos`](prepare/__main__.py) also read your own data. On `prepare lerobot`, set [`--whole-body`](prepare/lerobot.py) to `humanoid` or `legged_no_arms` for a robot that moves its whole body, and [`--robot`](prepare/lerobot.py) replaces the robot description in the metadata. `label` takes the harness options after [`--`](label/run.py), such as [`--max-tokens`](label/harness.py), [`--cell-w`](label/harness.py) and [`--max-spend`](label/harness.py).

A paid staged run requires a clean tracked checkout. [`--resume RUN --why "reason"`](label/run.py) continues it with its remaining budget on the same commit, and needs [`--dataset`](label/run.py), [`--episodes`](label/run.py), [`--kind`](label/run.py) and the original harness options again. [`--cap`](label/run.py) sets a new cap.

**Dataset adapters**

The public adapters are [`molmo`](prepare/molmo.py), [`abc130k`](prepare/abc130k.py), [`galaxea`](prepare/galaxea.py), [`habit`](prepare/habit.py), [`fastumi`](prepare/fastumi.py), [`realomin`](prepare/realomin.py), [`egocentric100k`](prepare/egocentric100k.py), [`genhumanego`](prepare/genhumanego.py), [`openaoe`](prepare/openaoe.py) and [`hiw500`](prepare/hiw500.py), and [`configs/slices/`](configs/slices/) records the published selections. The generic readers keep source fields and encoded timestamps, and do not apply a corpus release's cropping or task selection. This command prepares the MolmoAct2 episode from the quickstart and makes no model calls.

```console
uv run python -m prepare molmo prepare --episodes configs/quickstart/teleop.txt \
  --out data/episodes/molmo/quickstart
```

**Damaged data, clocks and units**

Argus reads each episode in a separate process. It leaves out a damaged state, signal, sidecar, metadata file or camera, names it on the episode, and keeps every camera it can still read. When one gripper or arm loses tracking for longer than 0.5s, as a dataset's `observation.state_valid` can mark, only that stretch of that actor has no reading, and the checks run on the rest. Where a column such as `left_chunk` says a gripper's pose moved to another SLAM map, no step is read across the switch. When a file hangs or crashes a reader, Argus reads the episode again with its cameras alone and names what that read skipped. It reads a damaged ZIP through both its central directory and its local headers, and opens nested archives up to 20,000 files and 40GB per upload.

Frame rates come from capture times, as the median frame step, or the rate over the recorded span when the two disagree by more than 2% ([`prepare/frame_rate.py`](prepare/frame_rate.py)). A camera whose timestamps disagree with its declared rate is placed by its timestamps and flagged. When capture clocks cannot be shared, the MCAP reader falls back to a recorder arrival clock, which does not establish physical synchronization. A field whose clock cannot be related to the footage is listed by its range rather than drawn on the timeline ([`prepare/signal_alignment.py`](prepare/signal_alignment.py)).

Argus names array elements from any declaration the recording carries ([`prepare/value_names.py`](prepare/value_names.py)) and marks the rest as unnamed. Joint angles declared in degrees, or undeclared ones whose readings can only be degrees, are converted to radians. A pose named x y z rx ry rz is read only when the dataset's documentation declares rotation vectors in metres and radians, since other arms use those names for Euler angles. A state declared in radians whose readings can only be degrees stays as recorded, is flagged, and is left out of kinematic checks. A publisher's declared layout, such as the [OpenTouch contract](configs/sensor_schemas/opentouch_hdf5_v1.json), is treated as a claim, not a verified map of the hardware.

**Custom boards**

Create [`data/boards/mine/manifest.json`](configs/quickstart/board.json) with the dataset, prepared episodes and run paths, then use [`python -m board build`](board/build.py), [`clips`](board/clips.py), [`serve`](board/serve.py), and [`static media`](board/static.py)/[`static site`](board/static.py), each with [`--help`](board/__main__.py). A rebuild reuses every unchanged episode unless you pass [`--no-cache`](board/build.py).

To add fields from a later pass without touching the run's output, put one file per episode in [`additions/`](board/build.py). An added field never overwrites one the label has, and it carries its provenance. A manifest can also name [`review_tags`](board/build.py), a JSON file of tags and reasons per episode, which join the rail's filter. [`--dictionary-job JOB`](board/serve.py) on [`board serve`](board/serve.py) edits field definitions in the local viewer.

The board answers only to `localhost`, and when [`--host`](board/serve.py) is a network address, also to IP addresses and the machine's own names. A reverse proxy that keeps the visitor's Host header gets 403 until you pass that name with [`--allow-host NAME`](board/serve.py). A board on 127.0.0.1 without `--allow-host` shows the uploader's metadata values in its pages, and its downloads leave them out.

Export a review as a static viewer by preparing the media first, then building the site, which is written to [`data/review/mine/static/<build_id>/`](board/static.py).

```console
uv run python -m board static media --board data/review/mine --clips data/review/mine/clips
uv run python -m board static site --board data/review/mine --clips data/review/mine/clips
```

**Review output and resuming**

A free review stops after the prepared [`episodes/`](review/__main__.py) and base requests in [`dry/`](review/__main__.py). Paid annotation adds model inputs and receipts in [`run/`](review/__main__.py), rejoined labels in [`run_final/`](review/__main__.py), and the board. `review` exits 1 when any unit failed, and still builds the board. The viewing copies `--board-clips off` skips are the most CPU a review spends, and without them the board has no video. After annotation, one text-only call picks at most two sensor findings to overlay at a time and rewrites every finding in plain English.

A resumed review refuses changed source data, inputs, or preparation and annotation settings, which need a fresh `--out`. After a code change, Argus verifies a recording labelled in one request against its saved receipt and rebinds the reply without another call.

**Retries**

A reply counts as labelled only when its timeline covers every sampled instant, it gives the outcome its request asks for, and every face issue gives a start and end ([`face_gaps`](label/completeness.py)). Argus resends a reply whose only gap is a face issue, and never retries another unusable model-written reply on its own. To pay for another attempt, add [`--retry-received EPISODE`](review/__main__.py) to the review command, optionally with a higher [`--max-tokens`](review/__main__.py). [`python -m label.reparse RUN`](label/reparse.py) parses stored replies again without a call.

A request refused with a transient status (408, 409, 425, 429, 500, 502, 503 or 529), or one that never left the machine, cost nothing and is tried up to four times. OpenRouter's refusal inside a 200 reply ([`REFUSED_IN_REPLY`](label/harness.py)) is handled as its status. Any other reply carrying the provider's error, or none, is kept, counted against the cap, and resent at most twice, after 30s and 120s.

**Cost caps**

A capped run uses verified model rates or a saved OpenRouter pricing quote and refuses unknown prices. Until a bill arrives, Argus reserves an upper bound from the request's text and images plus its full output limit, about $3.70 for a short teleop episode on the default model. A request killed before its reply keeps its reservation until the next run or [`--resume`](label/run.py) asks OpenRouter what it billed. Field interpretation and teleop image routing always use `openai/gpt-6-sol` and count against the cap. A cap refusal exits nonzero and keeps each unfinished unit for resume.

**Other checks**

[`checks.timebase scan`](checks/timebase.py) detects MolmoAct2 episodes that play faster than real time, and [`apply`](checks/timebase.py) adds its measurements and flags without changing their clocks. A check that stops with an error is reported as errored, never as clean. Optional hand pose uses [`board/hand_pose/modal_app.py`](board/hand_pose/modal_app.py) and [`board/hands.py`](board/hands.py), runs separately from annotation, and falls under the licenses below.

</details>

<details>
<summary>Model comparisons, gate cases and offline verification</summary>

```console
uv run python -m compare prepare --selection configs/compare/main.json
uv run python -m compare label --selection configs/compare/main.json --models astra,sol61_high --kind full --cap 20
uv run python -m compare board --entries data/runs/compare/main.json --out data/boards/compare
uv run python -m board build data/boards/compare
uv run python -m gate prepare
uv run python -m gate label --kind full --cap 20
```

Comparison caps apply per model run and gate caps per recording setup. Because sampling can differ by model, compare saved requests before claiming identical inputs. [`--with-example`](compare/__main__.py) supplies a reference annotation, and [`--routes`](compare/__main__.py) reuses resolution choices. [gate/selection.json](gate/selection.json) and [gate/cases.json](gate/cases.json) define the selected cases, and [`gate score`](gate/score.py) evaluates the supplied runs.

[`uv run --frozen --offline pytest -q -ra`](.github/workflows/ci.yml) runs the offline tests; viewer tests also require Node.js. [`tests/test_setup_matrix.py`](tests/test_setup_matrix.py) runs a generated recording of every supported setup through `review` and the export against an offline model, and fails on any recording, camera, signal, field or finding dropped without a recorded reason.

</details>

<details>
<summary>Every command's options</summary>

Each command runs as `python -m <command>`; `--help` describes its options.

| Command | Options |
|---|---|
| [`review`](review/__main__.py) | `--data`, `--rig`, `--out` and every option in [Settings](#settings) |
| [`review seam`](label/seam.py) | `--label`, `--footage`, `--out`, `--dry-run`, `--cap`, `--model`, `--reasoning`, `--concurrency`, `--timeout`, `--apply` |
| [`prepare abc130k prepare`](prepare/abc130k.py) | `--episodes`, `--out`, `--raw`, `--jobs`, `--force`, `--keep-mcap` |
| [`prepare abc130k sample`](prepare/abc130k.py) | `--list`, `--per-task`, `--split`, `--seed`, `--jobs` |
| [`prepare egocentric100k prepare`](prepare/egocentric100k.py) | `--episodes`, `--out`, `--raw`, `--jobs`, `--force` |
| [`prepare egocentric100k sample`](prepare/egocentric100k.py) | `--out`, `--list`, `--raw`, `--hours`, `--jobs`, `--seed` |
| [`prepare fastumi prepare`](prepare/fastumi.py) | `--episodes`, `--out`, `--raw`, `--jobs`, `--force` |
| [`prepare fastumi sample`](prepare/fastumi.py) | `--list`, `--raw`, `--per-task`, `--seed` |
| [`prepare folder prepare`](prepare/folder.py) | `--root`, `--rig`, `--out`, `--dataset`, `--max-minutes` |
| [`prepare galaxea prepare`](prepare/galaxea.py) | `--episodes`, `--out`, `--raw`, `--jobs`, `--force` |
| [`prepare galaxea sample`](prepare/galaxea.py) | `--out`, `--list`, `--raw`, `--hours`, `--jobs`, `--seed` |
| [`prepare genhumanego prepare`](prepare/genhumanego.py) | `--episodes`, `--out`, `--raw`, `--jobs`, `--force`, `--keep-mcap` |
| [`prepare hiw500 sample`](prepare/hiw500.py) | `--list`, `--raw`, `--hours` |
| [`prepare hiw500 prepare`](prepare/hiw500.py) | `--episodes`, `--out`, `--raw`, `--jobs`, `--force` |
| [`prepare habit prepare`](prepare/habit.py) | `--episodes`, `--out`, `--raw`, `--jobs`, `--force` |
| [`prepare lerobot prepare`](prepare/lerobot.py) | `--episodes`, `--out`, `--jobs`, `--force`, `--root`, `--rig`, `--dataset`, `--whole-body`, `--robot` |
| [`prepare molmo prepare`](prepare/molmo.py) | `--episodes`, `--out`, `--raw`, `--jobs`, `--force` |
| [`prepare molmo sample`](prepare/molmo.py) | `--list`, `--raw`, `--hours`, `--mode`, `--per-task-cap-hours` |
| [`prepare openaoe prepare`](prepare/openaoe.py) | `--episodes`, `--out`, `--raw`, `--jobs`, `--force` |
| [`prepare realomin prepare`](prepare/realomin.py) | `--episodes`, `--out`, `--raw`, `--jobs`, `--force`, `--keep-mcap` |
| [`prepare realomin sample`](prepare/realomin.py) | `--out`, `--list`, `--raw`, `--hours`, `--jobs`, `--seed`, `--keep-mcap` |
| [`prepare videos prepare`](prepare/videos.py) | `--episodes`, `--out`, `--jobs`, `--force`, `--root`, `--rig`, `--instructions`, `--dataset` |
| [`checks`](checks/__main__.py) | `--jobs`, `--force` |
| [`checks.capture_qc`](checks/capture_qc.py) | `--jobs`, `--force` |
| [`checks.sensors`](checks/sensors.py) | `--jobs`, `--force` |
| [`checks.stream_pairing`](checks/stream_pairing.py) | `--jobs`, `--force`, `--jumps`, `--grippers` |
| [`checks.timebase apply`](checks/timebase.py) | `--timebase`, `--labels` |
| [`checks.timebase scan`](checks/timebase.py) | `--raw`, `--out`, `--jobs` |
| [`label`](label/run.py) | `--dataset`, `--episodes`, `--kind`, `--cap`, `--runs`, `--concurrency`, `--note`, `--label`, `--resume`, `--why` |
| [`label.catalog`](label/catalog.py) | `--label`, `--out`, `--base`, `--cap`, `--model`, `--reasoning`, `--judge-model`, `--check`, `--concurrency`, `--dry-run` |
| [`label.harness`](label/harness.py) | `--episode-dir`, `--episodes-root`, `--out`, `--out-dir`, `--model`, `--reasoning`, `--sample-fps`, `--dataset-note`, `--inspection`, `--max-tokens`, `--timeout`, `--concurrency`, `--limit`, `--force`, `--retry-received`, `--cell-w`, `--example-dir`, `--resolution-route`, `--route-seeds`, `--max-spend`, `--dry-run`, `--allow-unverified-fees`, `--service-tier`, `--flex-fallback` |
| [`label.issue_groups`](label/issue_groups.py) | `--label`, `--dump`, `--out`, `--takes`, `--seed`, `--cap`, `--model`, `--reasoning`, `--concurrency`, `--dry-run` |
| [`board build`](board/build.py) | `--no-cache` |
| [`board clips`](board/clips.py) | `--episodes`, `--out`, `--only`, `--jobs`, `--clip-threads`, `--force`, `--decode-only`, `--name-prefix`, `--labels` |
| [`board export-lerobot`](board/lerobot_export.py) | `--dataset`, `--labels`, `--out`, `--report` |
| [`board grids`](board/grids.py) | `--jobs`, `--dataset`, `--file` |
| [`board hands`](board/hands.py) | `--src`, `--qa`, `--clips`, `--episodes`, `--out`, `--jobs` |
| [`board sensors`](board/sensors.py) | `--qa`, `--episodes`, `--out` |
| [`board serve`](board/serve.py) | `--board`, `--clips`, `--host`, `--port`, `--title`, `--header`, `--allow-host`, `--publish-uploader-values`, `--dictionary-job` |
| [`board static`](board/static.py) | `--board`, `--clips`, `--compare`, `--hands`, `--keypoints`, `--sensors`, `--out`, `--jobs`, `--threads`, `--limit-per-dataset`, `--dataset`, `--samples`, `--first`, `--public-base`, `--publish-uploader-values`, `--dictionary-job`, `--build-id`, `--force`, `--title`, `--header` |
| [`board to_board`](board/to_board.py) | `--in-dir`, `--out-dir`, `--dataset` |
| [`board.hand_pose.specs`](board/hand_pose/specs.py) | `--out` |
| [`compare board`](compare/__main__.py) | `--entries`, `--out` |
| [`compare label`](compare/__main__.py) | `--selection`, `--episodes`, `--models`, `--with-example`, `--kind`, `--cap`, `--runs`, `--concurrency`, `--routes` |
| [`compare prepare`](compare/__main__.py) | `--selection`, `--episodes` |
| [`compare.metrics`](compare/metrics.py) | `--json` |
| [`gate label`](gate/__main__.py) | `--episodes`, `--rigs`, `--kind`, `--cap`, `--runs`, `--concurrency` |
| [`gate prepare`](gate/__main__.py) | `--episodes` |
| [`gate score`](gate/score.py) | `--json` |

</details>

## Citation

If you use Argus or the labels it produces, please cite the [blog post](https://pantheon.inc/research/argus). GitHub's "Cite this repository" button reads the same entry from [`CITATION.cff`](CITATION.cff).

```bibtex
@article{li2026data2,
  author = {Eric Li},
  title = {Argus: A Dense Annotator for Robotics Data},
  journal = {Pantheon Research},
  year = {2026},
  month = {October},
  day = {1}
}
```

## Credits

The capture checks in [`checks/vendor/public_dataset_adapter_qc.py`](checks/vendor/public_dataset_adapter_qc.py) are Sambhav Gupta's, from public-dataset-adapter (commit 99d0a9e), with the changes listed in that file's header. The hand pose model is ACE-Ego-Hand (Yufei Liu et al., [arXiv:2608.20308](https://arxiv.org/abs/2608.20308)), built on Wan2.2, VideoX-Fun and MANO (Romero, Tzionas and Black, 2017).

## Licenses

The code is Apache-2.0 ([`LICENSE`](LICENSE)), and the few files that carry material under another license are listed in [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt). Full datasets and optional components are downloaded at run time; the architecture chart includes attributed image excerpts. The labels Pantheon publishes on the data dashboard, in its downloads and in the release are licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Anyone may use them for any purpose with credit to Pantheon. DiverseUMI-sample100, labels included, is released under the Pantheon Commercial Use License, Version 1.0. The footage they describe keeps its dataset's license.

<details>
<summary>Dataset and optional component licenses</summary>

| Component | License | Source |
|---|---|---|
| ACE-Ego-Hand code (commit 9757868) | MIT | [github.com/ggxxii/ACE-Ego-Hand](https://github.com/ggxxii/ACE-Ego-Hand/blob/97578680931b3f1c8111396c100d595b98857fb8/LICENSE) |
| ACE-Ego-Hand checkpoints ([`ace_ego_hand_k.pt`](https://huggingface.co/acerobotics2025/ACE-Ego-Hand), [`ace_ego_hand_kfree.pt`](https://huggingface.co/acerobotics2025/ACE-Ego-Hand)) | CC BY-NC 4.0, non-commercial | [huggingface.co/acerobotics2025/ACE-Ego-Hand](https://huggingface.co/acerobotics2025/ACE-Ego-Hand) |
| Wan2.2-Fun-5B-Control (weights, VAE, umT5-XXL text encoder and tokenizer) | Apache-2.0 | [huggingface.co/alibaba-pai/Wan2.2-Fun-5B-Control](https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-Control) |
| VideoX-Fun (commit 968f0e2) | Apache-2.0 | [github.com/aigc-apps/VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) |
| smplx 0.1.28 | Max Planck license for non-commercial scientific research | [github.com/vchoutas/smplx](https://github.com/vchoutas/smplx/blob/main/LICENSE) |
| MANO (you register and download it yourself) | Max Planck license for non-commercial scientific research, no redistribution | [mano.is.tue.mpg.de/license.html](https://mano.is.tue.mpg.de/license.html) |
| Python packages in [`uv.lock`](uv.lock) | Each package retains its own license | Package metadata and source distributions |
| MolmoAct2-BimanualYAM-Dataset (Ai2) | Apache-2.0 | [allenai/MolmoAct2-BimanualYAM-Dataset](https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset) |
| ABC-130k (XDOF) | Apache-2.0 | [XDOF/ABC-130k](https://huggingface.co/datasets/XDOF/ABC-130k) |
| Galaxea Open-World Dataset (Galaxea) | CC BY-NC-SA 4.0, non-commercial | [OpenGalaxea/Galaxea-Open-World-Dataset](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset); the adapter reads the copy at [`RogersPyke/Galaxea-Open-World-Dataset_10K_20260123`](https://huggingface.co/datasets/RogersPyke/Galaxea-Open-World-Dataset_10K_20260123); accept Galaxea's terms on its page before using it |
| HABIT (Config) | CC BY 4.0 | [configinc/HABIT](https://huggingface.co/datasets/configinc/HABIT) |
| FastUMI-100K | Apache-2.0 | [IPEC-COMMUNITY/FastUMI_100k_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/FastUMI_100k_lerobot) |
| 10Kh-RealOmin-OpenData (GenRobot) | CC BY-SA 4.0 | [genrobot2025/10Kh-RealOmin-OpenData](https://huggingface.co/datasets/genrobot2025/10Kh-RealOmin-OpenData) |
| Egocentric-100K (Build AI) | Apache-2.0 | [builddotai/Egocentric-100K](https://huggingface.co/datasets/builddotai/Egocentric-100K) |
| Gen-HumanEgo (GenRobot) | CC BY-SA 4.0 | [genrobot2025/Gen-HumanEgo](https://huggingface.co/datasets/genrobot2025/Gen-HumanEgo) |
| OpenAoE-2000h (inclusionAI) | Open-AoE Dataset License, attribution required; its MANO-derived files, which the adapter never reads, also fall under MANO's license | [inclusionAI/OpenAoE-2000h](https://huggingface.co/datasets/inclusionAI/OpenAoE-2000h/blob/main/LICENSE) |
| HIW-500 (BitRobot) | CC BY 4.0 | [BitRobot/HIW-500-LeRobot](https://huggingface.co/datasets/BitRobot/HIW-500-LeRobot) |

Running a dataset through this pipeline does not change its license. When you publish a dataset's footage or labels, credit it as its license asks. Because the hand keypoints come from the ACE-Ego-Hand checkpoints (CC BY-NC 4.0) and a model built on MANO (non-commercial research only), use them only for non-commercial purposes and credit ACE-Ego-Hand.

</details>
