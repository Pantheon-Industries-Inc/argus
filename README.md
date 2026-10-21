<h1 align="center">Argus</h1>

<p align="center">
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.11-455468" alt="Python 3.11"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-455468" alt="Apache-2.0"></a>
</p>

<table align="center">
  <tr>
    <td><strong><a href="https://pantheon.inc/research/argus">Blog</a> | <a href="https://pantheon.inc/data-board">Dashboard</a> | <a href="https://huggingface.co/datasets/Pantheon-inc/DiverseUMI-sample100">DiverseUMI-100</a></strong></td>
  </tr>
</table>

> [!IMPORTANT]
> Argus has many tunable command-line options. Please read [Settings](#settings) in full.

### \* DiverseUMI (October 2026)

We released [DiverseUMI](https://huggingface.co/datasets/Pantheon-inc/DiverseUMI-sample100), 99.6 hours of bimanual UMI annotated by Argus at Sol 6.1 High. It spans 12,219 episodes across 179 skills, each with dense action, outcome and progress labels, key events, object state changes, scene graphs and operator recoveries, alongside SLAM-tracked wrist poses and gripper actions.

### \* WGO-Bench (October 2026)

Recently, [Midcentury AI](https://www.midcentury.xyz/blog/optimizing-argus) pushed an adaptation of Argus to SOTA (**50.2% semantic F1**) on WGO-Bench. However, this result uses **Sol 6.1 High**, so its advantage over older-model baselines cannot be attributed to the harness alone.

### \* Argus improvements (October 2026)

Argus now annotates depth, tactile and other sensor data jointly with video, state and actions. By default the model inspects any recording with depth or a touch sensor, reading any of its fields at full rate, and any recording with fields the prompt does not carry, such as EgoDex hand poses. A recording with only arm signals and a wrist force/torque sensor is inspected only with [`--inspection all`](#settings), and its fields still reach the model as sampled readings.
Argus also scores each episode out of 10 for execution and outcome, and each subgoal for execution, deterministically from the annotations. Outcome is whether the goal was reached and how far progress got; execution is a weighted mean of cleanliness, steadiness and speed, which [`board/scores.py`](board/scores.py) defines as cleanliness, fluency and efficiency. A subgoal is a lasting state change the goal depends on, such as a block placed in the row, never a pick or a grasp.

## Overview

To my knowledge, Argus is the most advanced open-source robot learning annotation harness (as of 2026).

It is the only harness to combine native multi-format ingestion (rather than restricting inputs eg to LeRobot, as [LeRobot's annotation pipeline](https://huggingface.co/docs/lerobot/main/annotation_pipeline) does), joint visual/sensor reasoning, dense annotations of **actions, subgoals, outcomes, operator mistakes, recoveries and object-state changes**, and execution and outcome scores for each episode and subgoal. It reads multi-view RGB alongside state/action trajectories, depth, tactile and other recorded fields, preserving their timing and measurement semantics. For depth and tactile data the VLM can inspect additional samples or image regions to support a finding; evidence, confidence and limitations remain available in the viewer and JSON exports.

<p align="center">
  <a href="https://raw.githubusercontent.com/Pantheon-Industries-Inc/argus/refs/heads/main/media/architecture.svg?v=a4ebb130d873">
    <img src="media/architecture.svg?v=a4ebb130d873" width="1900" alt="Annotation pipeline: format ingestion, field interpretation (on by default), visual, state and sensor evidence, evidence inspection (automatic for depth and touch recordings, and for fields the prompt does not carry), VLM annotation of behavior, state changes and sensor findings with a separate regional pass for tactile arrays, issue detection, temporal stitching of request parts and session tasks, and a viewer with chosen sensor cards, quality scores and JSON exports.">
  </a>
</p>

| Input support | Details |
|---|---|
| Recording types | Teleop, UMI and human ego |
| Native formats | LeRobot v2.0/v2.1/v3.0, MCAP, HDF5, video files and folders |
| Archives | ZIP and TAR, plain or compressed with gzip, bzip2 or xz |
| Sidecar metadata | JSON, JSON Lines, CSV, TSV, text and Markdown notes, and LeRobot's Parquet and JSON Lines annotation tables |
| Public dataset adapters | Nine, each downloading and preparing a public dataset |

| Available evidence | How it contributes |
|---|---|
| RGB | Timestamped multi-view frames provide visual context for task execution, object interactions, progress, failures and recoveries. |
| Tactile, inspected by default | Pressure, normal/shear force, contact arrays and optical tactile images support analysis of grip, contact distribution and surface motion. Full-array patterns, individual-cell and regional responses, and hand geometry are compared with video and gripper state. Optical tactile images are compared at the moments they change most, and arrays with regional evidence get a separate regional pass. Array shape alone does not establish anatomy or force, and response changes alone do not establish slip. |
| Depth, inspected by default | Metric depth supports camera-relative distance and separation when units, calibration and temporal alignment are known. Relative depth remains relative. |
| Proprioception and actions | Joint states, end-effector poses, gripper states and action trajectories provide motion context. Training targets retain their declared semantics rather than being assumed to be executed commands. |
| Other episode fields | Force/torque, numeric channels, event flags, waveform packets and structured metadata reach the prompt as range lines. A channel the sampling rate resolves also gets a reading at each sampled instant, and one that tracks another channel is read only where it departs from it. A channel the sampling rate does not resolve gets its mean over each sampling window while a small share of the request allows, and otherwise the times of its largest and smallest readings. [`--inspection all`](label/policy.py) also inspects a recording whose only extra fields are these. |
| Task context | Instructions, annotations and calibration metadata provide context. The VLM checks publisher claims against the recording rather than assuming their correctness. |

## Install and review

You need Python 3.11, [uv](https://docs.astral.sh/uv/), and [`ffmpeg`](https://ffmpeg.org/ffmpeg.html)/[`ffprobe`](https://ffmpeg.org/ffprobe.html) 5.1 or newer.

```console
git clone https://github.com/Pantheon-Industries-Inc/argus.git
cd argus
uv sync --frozen
```

Paid annotation requires [`OPENROUTER_API_KEYS`](label/harness.py) (comma-separated keys) or [`OPENAI_API_KEY`](label/harness.py) in your shell or secret manager. Gated Hugging Face datasets require accepting their terms and setting [`HF_TOKEN`](prepare/hub.py).

Annotate a dataset under a spending cap, then open its board. Add [`--free`](review/__main__.py) to build the same requests without model calls.

```console
uv run python -m review --data path/to/dataset --rig teleop_arms \
  --dataset-note "Two arms arrange blocks on a tray." --out data/review/mine --cap 20
uv run python -m board serve --board data/review/mine --clips data/review/mine/clips
```

[`--data`](review/__main__.py) accepts a folder, file, archive or HTTP(S) URL. Set [`--rig`](review/__main__.py) to [`teleop_arms`](review/__main__.py), [`handheld_gripper`](review/__main__.py) or [`ego_head`](review/__main__.py) for teleop, UMI or human ego, respectively. Argus detects the format and the evidence available, and RGB-only data runs through the same pipeline. [Data Review](https://data.pantheon.inc/review) uses the same default policy, though its deployed version and upload limits can differ.

### Settings

| Setting | Default | Options |
|---|---|---|
| [`--model`](label/policy.py) | `openai/gpt-6-astra` | `openai/gpt-6.1-sol` (Sol 6.1), or `google/gemini-4-argon` (Gemini 4 Argon) with an OpenRouter key, priced from OpenRouter's published rates ([`label/pricing.py`](label/pricing.py)) |
| [`--reasoning`](label/policy.py) | Medium for Astra, high for every other model | `none`, `minimal`, `low`, `medium`, `high` or `xhigh` |
| [`--sample-fps`](label/policy.py) | A frame every 1.5s for teleop, 1s for UMI and 0.5s for human ego, plus the edges of still spans and the first and last frames; Sol samples every 1s on every rig, human ego included, and refuses sparser sampling | `2` catches subsecond fumbles and rapid recoveries, at the cost of more images and, on teleop and UMI, shorter request windows |
| [`--task-mode`](label/policy.py) | From source metadata, otherwise `episode` | `session` labels the recording as a sequence of tasks, each with its own outcome, which suits freeform recordings and repeated cycles; human ego uses the activities schema. In either mode a recording with an instruction is checked against it |
| [`--inspection`](label/policy.py) | `auto`, for recordings with depth or a touch sensor, or with fields the prompt does not carry | `all` also inspects recordings with no depth or touch sensor, such as one with only force/torque, arm signals, actions or source metadata; `off` makes no evidence selection call, though a tactile array still gets its separate regional pass |
| [`--field-dictionary`](review/__main__.py) | `on` | `off` skips the one call per upload (at most $3) that asks a model what unclear recorded fields mean; fields are then labelled as recorded |
| [`--cap`](review/__main__.py) | $20 | The most the job may spend, counting every call and outstanding reservation |
| [`--max-minutes`](review/__main__.py) | All footage | Reads at most this many minutes of footage |
| [`--dataset`](review/__main__.py) | The name of `--data` | The name shown on the board and told to the model |
| [`--dataset-note`](label/policy.py) | The source's own collection note, if any | A description of the recordings given to the model as context |
| [`--max-tokens`](review/__main__.py) | 64,000 | The output limit of each labelling request; a smaller limit reserves less against the cap |
| [`--timeout`](review/__main__.py) | 600s | How long a labelling request waits for the provider's reply |
| [`--concurrency`](review/__main__.py) | 8 | Labelling requests in flight at once |
| [`--jobs`](review/__main__.py) | 2 | Decode processes per preparation step |
| [`--grouping`](review/__main__.py) | Inferred from file names | JSON such as `{"cam_folder": "cameras"}` or `{"take_folder": "takes"}`, for folders whose file names cannot tell separate takes from cameras of one take |
| [`--free`](review/__main__.py) | Off | Builds the base requests and stops before any model call |
| [`--force`](review/__main__.py) | Off | Relabels every selected episode with new paid calls, keeping earlier replies and charges |
| [`--retry-received EPISODE`](review/__main__.py) | None | Sends that episode's request again after an unusable reply the model wrote, or after a provider failure that has used its two automatic resends; repeat for each episode |

Each recording produces one episode. A recording longer than one request allows (450s at default sampling, less at denser sampling) is cut at still moments, labelled in parts and rejoined on the original timeline. Each part is checked against the recording's instruction on what that part shows. In episode mode the whole-recording outcome stays unclear, because no part sees both its start and its end, so use `session` for recordings of repeated or separate tasks. Rerunning a review with the same [`--out`](review/__main__.py) folder resumes it and reuses every output whose inputs still match. [`--free`](review/__main__.py) previews the base requests only, without field interpretation, routing, inspection or sensor cards.

## Results and further use

Live and static boards share one [viewer](board/index.html), which overlays behavioral and sensor annotations on the footage with the supporting measurements beside it. The same findings export as episode JSON, filtered JSON Lines and static sites. Issues with missing or invalid severity remain visible as unrated and are excluded from severity totals. For a run still in progress, give a dataset's manifest entry a [`target`](board/build.py) such as `{"episodes": 2682, "hours": 101.9}`, and the board shows the episodes and hours labelled so far against it, from before that dataset's first episode. A page open on `board serve` checks every 30s for a rebuild and updates its list and totals in place, keeping the open episode.

This is an excerpt from a pressure glove episode's downloadable JSON. In the list and on the video, a finding shows its `video_card` words when it has one, otherwise its `list_copy`, otherwise its own headline and detail. Evidence IDs resolve to recorded streams and samples in the full export:

```json
{
  "sensor_evidence": {
    "findings": [
      {
        "start_s": 2.63,
        "end_s": 3.03,
        "headline": "Thumb loading likely eases more than index finger loading",
        "detail": "As the cup moves away, the thumb response weakens substantially while the index finger response remains elevated.",
        "confidence": "medium",
        "video_card": {
          "version": 1,
          "headline": "The thumb likely presses less.",
          "detail": "The index finger keeps pressing as the cup moves away."
        },
        "evidence": [
          {
            "sensor_id": "mechanics:signal:right_pressure",
            "series_id": "mechanics:signal:right_pressure:derivation:regional_response:0",
            "time_s": [2.63, 2.83, 2.93]
          }
        ]
      }
    ]
  }
}
```

Public exports, meaning the static site, downloads and [`/api/export`](board/serve.py), name the uploader's own metadata (file attributes, sidecar fields, publisher fields) but leave out its values, because no filter can recognize every secret a person writes there. Add [`--publish-uploader-values`](board/serve.py) to `board serve` or `board static site` to publish them. Exports also redact access attributes, recognizable credential values and private access locations in every field.

Field definitions supplied by the dataset publisher or user override inferred meanings and layouts without changing recorded measurements. Changes to source data or field interpretations invalidate stale findings.

![Two-camera UMI annotation walkthrough, with action phases, subgoals and goal reasoning over the footage.](media/board_demo.webp)

[Browse example annotations](https://pantheon.inc/data-board).

<details>
<summary>Individual stages, dataset adapters and custom boards</summary>

```console
uv run python -m prepare folder prepare --root path/to/data --rig teleop_arms --out data/episodes/mine/all
uv run python -m checks data/episodes/mine/all
uv run python -m label --dataset mine --episodes data/episodes/mine/all --kind dry
uv run python -m label --dataset mine --episodes data/episodes/mine/all --kind full --cap 20
```

The staged commands label each prepared episode in one request: they do not split long recordings or interpret fields, so a recording longer than about 450s at default sampling needs `review`. Besides `prepare folder`, [`prepare lerobot`](prepare/__main__.py) and [`prepare videos`](prepare/__main__.py) read your own data. In `review`, a URL download is limited to 20 GiB and 24 hours. The staged commands list their own options with `--help`, and `label` passes the harness options after [`--`](label/run.py), such as [`--max-tokens`](label/harness.py), [`--timeout`](label/harness.py), [`--cell-w`](label/harness.py) for the grid cell width and [`--max-spend`](label/harness.py) for a remaining allowance.

Public adapters are [`molmo`](prepare/molmo.py), [`abc130k`](prepare/abc130k.py), [`galaxea`](prepare/galaxea.py), [`habit`](prepare/habit.py), [`fastumi`](prepare/fastumi.py), [`realomin`](prepare/realomin.py), [`egocentric100k`](prepare/egocentric100k.py), [`genhumanego`](prepare/genhumanego.py) and [`openaoe`](prepare/openaoe.py); [`configs/slices/`](configs/slices/) records published selections. Generic readers preserve source fields and encoded timestamps, using publisher metadata to align episode frames and sensor samples. MCAP falls back to a common recorder arrival clock when capture clocks cannot be shared; arrival timing does not establish physical synchronization. Fields whose clocks cannot be related to the footage remain available as untimed source evidence, except an episode's own sensor file, which is placed from both starts and marked as an assumption. Generic readers do not apply a corpus release's cropping or task selection. Raw video packets with frame reordering are rejected when decode timestamps are unavailable; other readable cameras are retained. When part of an episode is damaged or unreadable, such as its state, a signal, its metadata, a sidecar or one camera, that part is left out and named on the episode, and the episode keeps every camera that can still be read. Each episode is read in a separate process, so a file that hangs or crashes a native reader is read again with its cameras alone instead of stopping the upload. A camera whose own timestamps disagree with the declared frame rate is placed by its timestamps, and the disagreement is flagged. A frame rate whose median step is a stamp a few microseconds after the one before is measured over the frames' whole span instead, and an episode whose stated rate its frames' capture times contradict more than tenfold is labelled at the frames' own rate, with the stated one named in its checks. A video used as it was uploaded is decoded once while it is prepared, so frames the decoder hides after damage are named with their span. A damaged ZIP is read through both of its indexes, and a member whose two names disagree is read as an upload of its own, so no table, task or metadata of the upload it came from describes it. Archives inside archives are opened at any depth, up to 20,000 files and 40 GB for the whole upload, and past either limit the unpack stops and says so. A member the filesystem cannot hold is left out and named while the rest of the upload is read.

Every reader (LeRobot, HDF5 and MCAP) converts joint angles declared in degrees, or whose readings can only be degrees, to radians, and the state contract records why. A state declared in radians whose readings can only be degrees is kept as recorded, flagged, and left out of kinematic checks.

A publisher's declared layout, such as the bundled [OpenTouch contract](configs/sensor_schemas/opentouch_hdf5_v1.json) that names the glove's hand regions, is treated as the publisher's claim, not a verified map of the hardware.

For example, download and prepare the single MolmoAct2 episode listed in the quickstart:

```console
uv run python -m prepare molmo prepare --episodes configs/quickstart/teleop.txt \
  --out data/episodes/molmo/quickstart
```

The adapter retrieves the source files needed for that episode, including shared video packs, and preserves its camera offsets and state/action trajectories. Preparation makes no model calls.

For a custom board, create [`data/boards/mine/manifest.json`](configs/quickstart/board.json) with the dataset, prepared episodes and run paths. [configs/quickstart/board.json](configs/quickstart/board.json) shows the structure with published-dataset severity rules. Use [`python -m board build`](board/build.py), [`clips`](board/clips.py), [`serve`](board/serve.py), and [`static media`](board/static.py)/[`static site`](board/static.py); each takes [`--help`](board/__main__.py). A rebuild reuses each episode whose run output, prepared folder, manifest and code are unchanged, so rebuilding a large board during a run stays quick; [`--no-cache`](board/build.py) builds every episode again. Refresh the board after changing field definitions. To edit them in the local viewer, add [`--dictionary-job JOB`](board/serve.py) to [`board serve`](board/serve.py). The board answers only to `localhost`, and when [`--host`](board/serve.py) is a network address, also to IP addresses and the machine's own names. A reverse proxy that keeps the visitor's Host header, such as `tailscale serve` or Caddy, therefore gets 403 until you pass that name with [`--allow-host NAME`](board/serve.py). A board served on 127.0.0.1 without `--allow-host` shows those values in its pages, and its downloads still leave them out.

Export the same review as a static viewer by preparing the media first, then building the site:

```console
uv run python -m board static media --board data/review/mine --clips data/review/mine/clips
uv run python -m board static site --board data/review/mine --clips data/review/mine/clips
```

The site is written to [`data/review/mine/static/<build_id>/`](board/static.py), with shared media in [`data/review/mine/static/media/`](board/static.py).

A resumed review keeps its settings, refuses changed ones, and reuses its verified preparation, so use a fresh `--out` to change source data, prepared inputs or annotation settings, or to apply newer reader fixes. After a code change, a recording labelled in one request is verified offline against its saved receipt, exact prompt and image order included, and its retained reply is rebound without another call; when that proof is missing or has changed, [`--force`](label/harness.py) relabels the episode with new paid calls. Unresolved charges stay reserved, and inspection blocked by the cap resumes when the allowance increases, before final annotation is dispatched.

A free review stops after prepared [`episodes/`](review/__main__.py) and base requests in [`dry/`](review/__main__.py). Paid annotation adds model inputs and receipts in [`run/`](review/__main__.py), rejoined labels in [`run_final/`](review/__main__.py), and the board; `review` exits 1 when any unit failed, and the board is still built.

After annotation, one text-only call to the same model picks the sensor findings worth overlaying on the video, at most two at once, and rewrites every finding in plain English; every finding stays in the list and the JSON export, and if the call fails, every finding plays on the video.

Paid staged runs require a clean tracked checkout. [`--resume RUN --why "reason"`](label/run.py) retains the run and remaining budget, using the same commit, dataset, prepared episodes and run kind; add [`--cap`](label/run.py) to set a new cap, for example to finish a run its cap stopped. Pass [`--dataset`](label/run.py), [`--episodes`](label/run.py) and [`--kind`](label/run.py) again, and the original harness options after [`--`](label/run.py). Cross-host writers require working shared filesystem locks.

[`python -m label.reparse RUN`](label/reparse.py) retries parsing stored replies that did not parse and rebinds their evidence without another call. A reply the model wrote that is unusable, including truncated output, is never retried automatically. Add [`--retry-received EPISODE`](review/__main__.py) to the original review command to authorize another paid attempt for that episode, optionally raising [`--max-tokens`](review/__main__.py). For a staged resume, put these flags after [`--`](label/run.py), with one retry flag per episode. Healthy outputs and prior billing receipts are retained. A request refused with a transient error status (408, 409, 425, 429, 500, 502, 503 or 529), or one that never left the machine, cost nothing and is tried up to four times; any other request the provider refused unpaid is sent again by a plain resume. A reply that carries the provider's own error, such as an overloaded upstream, is retried the same way when it cost nothing. When the provider fails a request that may have been billed, because the reply carries its error or never arrives in full, Argus keeps that attempt and its possible charge, which stays counted against the cap, and sends the request again automatically, at most twice, after 30s and then 120s. A plain resume does the same for a unit sent fewer than three times. Past that, only `--retry-received` for that episode or `--force` sends it again. [`--force`](label/harness.py), also available in `review`, refreshes every selected episode, including completed ones. A cap refusal returns a nonzero exit status and retains each unfinished unit for resume.

Capped runs use verified model rates or a saved OpenRouter pricing quote and refuse unknown prices. Each request reserves an upper bound computed from the request itself, its text and images plus its full output limit, until the bill arrives. With the default 64,000-token output limit, the default model reserves at least $3.20 per request for the output alone, about $3.70 for a short teleop episode, so a smaller [`--max-tokens`](review/__main__.py) reserves less. Field interpretation and teleop image routing always use `openai/gpt-6-sol`, whatever `--model` says, and count against the cap; field interpretation can spend up to $3. When the cap cannot cover a single labelling request, `review` stops before that request and says so, though field interpretation may already have run. A quoted model whose price list includes web search is refused unless it is a Claude or Gemini model, which search only when a request asks, and requests here never ask; a price for an input a request never carries, such as audio, does not block a quote. Receipts preserve prior responses, charges and outstanding reservations across failures. [`board grids`](board/grids.py) reconstructs base request grids only when original input hashes can be verified; detail/sensor images are counted but not reconstructed there.

[`checks.timebase scan`](checks/timebase.py) detects MolmoAct2 episodes that play faster than real time. [`apply`](checks/timebase.py) adds its measurements and report flags to prepared episodes and existing labels without changing their clocks. A capture, source or sensor check that stops with an error is reported as errored, never as clean. Optional hand pose uses [`board/hand_pose/modal_app.py`](board/hand_pose/modal_app.py) and [`board/hands.py`](board/hands.py); it is separate from ordinary annotation and has the licenses below.

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

Comparison caps apply per model run and gate caps per recording setup. Sampling may differ by model, so compare saved requests before claiming identical inputs. [`--with-example`](compare/__main__.py) supplies a reference annotation; [`--routes`](compare/__main__.py) reuses resolution choices. [gate/selection.json](gate/selection.json) and [gate/cases.json](gate/cases.json) define the selected cases; [`gate score`](gate/score.py) evaluates the supplied runs. Its text checks count a word only where its own sentence asserts it, which is not a test of meaning; ambiguous object claims count as unmet and are flagged for review.

Comparisons distinguish missing responses from replies that failed to parse. Cost reports separate verified charges from unresolved reservations, and a gate cost rate is reported only when every invoice is verified. Cold-cost estimates require the default model and a single paid attempt per unit.

[`uv run --frozen --offline pytest -q -ra`](.github/workflows/ci.yml) runs offline tests; viewer tests also require Node.js. CI runs the same command as eight shards side by side, each a set of whole test files balanced by the times in [`tests/shard_times.json`](tests/shard_times.json); set [`ARGUS_TEST_SHARD=1/8`](tests/sharding.py) to run CI's first shard locally. CI uses the locked environment, Node, ffmpeg and ffprobe without provider credentials or dataset access. Secret scans cover the committed tree and reachable history.

</details>

## Credits

The capture checks in [`checks/vendor/public_dataset_adapter_qc.py`](checks/vendor/public_dataset_adapter_qc.py) are Sambhav Gupta's, from public-dataset-adapter (commit 99d0a9e), with the changes listed in that file's header. The hand pose model is ACE-Ego-Hand (Yufei Liu et al., [arXiv:2608.20308](https://arxiv.org/abs/2608.20308)), built on Wan2.2, VideoX-Fun and MANO (Romero, Tzionas and Black, 2017).

## Licenses

The code is Apache-2.0 ([`LICENSE`](LICENSE)), and the few files that carry material under another license are listed in [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt). Full datasets and optional components are downloaded at run time; the architecture chart includes attributed image excerpts. The labels Pantheon publishes, on the data dashboard, in its downloads and in the release, are licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), so anyone may use them for any purpose with credit to Pantheon. DiverseUMI, labels included, is released under the Pantheon Commercial Use License, Version 1.0. The footage they describe keeps its dataset's license.

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

Running a dataset through this pipeline does not change its license, so credit each dataset as its license asks when you publish its footage or labels. The hand keypoints come from the ACE-Ego-Hand checkpoints (CC BY-NC 4.0) and a model built on MANO (non-commercial research only), so use them only for non-commercial purposes and credit ACE-Ego-Hand.

</details>
