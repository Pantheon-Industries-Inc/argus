<h1 align="center">Argus</h1>

<p align="center">
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.11-455468" alt="Python 3.11"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-455468" alt="Apache-2.0"></a>
</p>

<table align="center">
  <tr>
    <td><strong><a href="https://pantheon.inc/research/argus">Blog</a> | <a href="https://pantheon.inc/data-board">Data</a></strong></td>
  </tr>
</table>

### \* WGO-Bench (October 2026)

Recently, [Midcentury AI](https://www.midcentury.xyz/blog/optimizing-argus) pushed an adaptation of Argus to SOTA (**50.2% semantic F1**) on WGO-Bench. However, this result uses **Sol 6.1 High**, so its advantage over older-model baselines cannot be attributed to the harness alone.

### \* Depth and tactile annotations (October 2026)

Argus now annotates depth, tactile and other sensor data jointly with video, state and actions. Sensor findings appear as video overlays and in the JSON export, with supporting evidence and uncertainty.

## Overview

To my knowledge, Argus is the most advanced open-source robot learning annotation harness (a/o 2026).

It is the only harness to combine native multi-format ingestion (rather than restricting inputs eg to LeRobot, as [LeRobot's annotation pipeline](https://huggingface.co/docs/lerobot/main/annotation_pipeline) does), joint visual/sensor reasoning, and dense annotations of **actions, subgoals, outcomes, operator mistakes, recoveries and object-state changes**. It reads multi-view RGB alongside state/action trajectories, depth, tactile and other recorded fields, preserving their timing and measurement semantics. The VLM can inspect additional samples or image regions to support a finding; evidence, confidence and limitations remain available in the viewer and JSON exports.

<p align="center">
  <a href="https://raw.githubusercontent.com/Pantheon-Industries-Inc/argus/refs/heads/main/media/architecture.svg?v=6fb8f2e48081">
    <img src="media/architecture.svg?v=6fb8f2e48081" width="1900" alt="Annotation pipeline: format ingestion, field interpretation, visual, state and sensor evidence, VLM annotation of behavior and state changes, issue detection, temporal stitching, viewer and JSON exports.">
  </a>
</p>

| Input support | Details |
|---|---|
| Recording types | Teleop, UMI and human ego |
| Native formats | LeRobot v2.0/v2.1/v3.0, MCAP, HDF5, video files and folders |
| Archives | ZIP and TAR |
| Sidecar metadata | JSON, CSV and Parquet |
| Public dataset adapters | Nine adapters for downloading datasets and selecting episode samples |

| Available evidence | How it contributes |
|---|---|
| RGB | Timestamped multi-view frames provide visual context for task execution, object interactions, progress, failures and recoveries. |
| Proprioception and actions | Joint states, end-effector poses, gripper states and action trajectories provide motion context. Training targets retain their declared semantics rather than being assumed to be executed commands. |
| Depth | Metric depth supports camera-relative distance and separation when units, calibration and temporal alignment are known. Relative depth remains relative. |
| Tactile | Pressure, normal/shear force, contact arrays and optical tactile images support analysis of grip, contact distribution and surface motion. Raw sensor response alone does not establish physical force or slip. |
| Other episode fields | Numeric channels, event flags, waveform packets and structured metadata are accessible through the same inspection interface. Spectral analysis of eligible waveforms requires a recorded sample rate. |
| Task context | Instructions, annotations and calibration metadata provide context. The VLM checks publisher claims against the recording rather than assuming their correctness. |

## Install and review

You need Python 3.11, [uv](https://docs.astral.sh/uv/), and [`ffmpeg`](https://ffmpeg.org/ffmpeg.html)/[`ffprobe`](https://ffmpeg.org/ffprobe.html) 5.1 or newer.

```console
git clone https://github.com/Pantheon-Industries-Inc/argus.git
cd argus
uv sync --frozen
```

Paid annotation requires [`OPENROUTER_API_KEYS`](label/harness.py) (comma-separated keys) or [`OPENAI_API_KEY`](label/harness.py) in your shell or secret manager. Gated Hugging Face datasets require accepting their terms and setting [`HF_TOKEN`](prepare/hub.py). Keep inputs and outputs under the ignored [`data/`](.gitignore#L1) directory.

Prepare and inspect base requests without model calls.

```console
uv run python -m review --data path/to/dataset --rig teleop_arms \
  --dataset-note "Two arms arrange blocks on a tray." --out data/review/mine --free
```

Annotate the same job under a spending cap, then open its board.

```console
uv run python -m review --data path/to/dataset --rig teleop_arms \
  --dataset-note "Two arms arrange blocks on a tray." --out data/review/mine --cap 20
uv run python -m board serve --board data/review/mine --clips data/review/mine/clips
```

[`--data`](review/__main__.py) accepts a folder, file, archive or HTTP(S) URL. Set [`--rig`](review/__main__.py) to [`teleop_arms`](review/__main__.py), [`handheld_gripper`](review/__main__.py) or [`ego_head`](review/__main__.py) for teleop, UMI or human ego, respectively. Argus detects the format and available evidence; RGB-only data follows the same pipeline without requiring sensor inputs. [Data Review](https://data.pantheon.inc/review) uses the same default policy, though its deployed version and upload limits can differ.

| Default or optional setting | Behavior |
|---|---|
| Astra, the default | [`openai/gpt-6-astra`](label/policy.py), medium reasoning; a regular sample every 1.5s for teleop, 1s for UMI and 0.5s for human ego, plus eligible boundaries and first/last frames. |
| Sol 6.1 High | [`--model openai/gpt-6.1-sol --reasoning high`](label/policy.py); defaults to 1 FPS on all recording setups and requires at least 1 FPS. |
| Faster actions | [`--sample-fps 2`](label/policy.py) is recommended for subsecond fumbles and rapid recoveries. More images shorten the available request window and can increase cost. |
| Task scope | Episode/session scope follows source metadata; [`--task-mode episode`](label/policy.py) or [`session`](label/policy.py) overrides it. A missing instruction does not automatically imply a session. Human ego uses the activities schema. |

Each recording produces one public episode. Long recordings are split to fit API request limits and rejoined on the original timeline; internal chunk boundaries are not treated as recording truncation. Session mode annotates activities within the recording. Preparation limits, missing chunks and processing failures are reported alongside any usable results.

Rerunning a review job with the same [`--out`](review/__main__.py) preserves its settings and reuses outputs whose inputs and selected evidence still match. Source data, prepared inputs and annotation settings must still match; otherwise start a new job. After code changes, saved request receipts can verify the exact prompt and ordered images offline and rebind the retained reply without another call. Missing or changed proof requires an explicit refresh. The preparation receipt retains the original reader version. Resume reuses that verified preparation; use a fresh `--out` to apply newer reader fixes. [`--free`](review/__main__.py) skips paid field interpretation, image-detail routing and evidence selection, so it previews base requests rather than the full paid evidence path. Spending caps account for all applicable calls and outstanding reservations; unresolved charges stay reserved. Inspection blocked by the cap resumes when the allowance increases, before final annotation is dispatched.

For models without locally verified rates, reservations use the full published context window and can exceed the expected request cost. Capped runs reject native search models whose search fees have no enforceable ceiling.

## Results and further use

Live and static boards use the same [viewer](board/index.html), with behavioral and sensor annotations overlaid on the footage and supporting measurements alongside it. The same findings are available as episode JSON, filtered JSON Lines and static exports. Issues with missing or invalid severity remain visible as unrated and are excluded from severity totals. Each issue's problem family is read from the grammar of its own sentences, so a problem the text denies, asks about, supposes or withdraws is not counted, and a correction in a later sentence leaves the earlier finding in its family.

This is an excerpt from a tactile episode's downloadable JSON. Evidence IDs resolve to recorded streams and samples in the full export:

```json
{
  "sensor_evidence": {
    "findings": [
      {
        "start_s": 54.0,
        "end_s": 55.0,
        "headline": "The final module's contact reshapes before the grip opens",
        "overlay_detail": "The pad deforms while the jaw reading stays steady; this does not establish that the module slipped.",
        "confidence": "medium",
        "evidence": [
          {
            "sensor_id": "camera:extra4",
            "time_s": [53.0, 54.0, 55.0]
          }
        ]
      }
    ]
  }
}
```

A free review job stops after prepared [`episodes/`](review/__main__.py) and base requests in [`dry/`](review/__main__.py). Paid annotation adds model inputs and receipts in [`run/`](review/__main__.py), rejoined labels in [`run_final/`](review/__main__.py), and the board. Action phases distinguish advancing, wasteful and idle behavior. Sensor findings include evidence, confidence and alternative explanations; inconclusive evidence need not produce a finding. Public exports, meaning the static site, downloads and [`/api/export`](board/serve.py), name the uploader's own metadata (file attributes, sidecar fields, publisher fields) but leave out its values, because no filter can recognize every secret a person writes there. A board served on 127.0.0.1 shows those values to the person running it. Add [`--publish-uploader-values`](board/serve.py) to `board serve` or `board static site` to publish them. Exports also redact access attributes, recognizable credential values and private access locations in every field.

Field definitions supplied by the dataset publisher or user override inferred meanings and layouts without changing recorded measurements. Changes to source data or field interpretations invalidate stale findings. Shared footage retains its source license and access policy.

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

Public adapters are [`molmo`](prepare/molmo.py), [`abc130k`](prepare/abc130k.py), [`galaxea`](prepare/galaxea.py), [`habit`](prepare/habit.py), [`fastumi`](prepare/fastumi.py), [`realomin`](prepare/realomin.py), [`egocentric100k`](prepare/egocentric100k.py), [`genhumanego`](prepare/genhumanego.py) and [`openaoe`](prepare/openaoe.py); [`configs/slices/`](configs/slices/) records published selections. Generic readers preserve source fields and encoded timestamps, using publisher metadata to align episode frames and sensor samples. MCAP falls back to a common recorder arrival clock when capture clocks cannot be shared; arrival timing does not establish physical synchronization. Fields whose clocks cannot be related to the footage remain available as untimed source evidence. Corpus-release cropping and task selection are not applied. Raw video packets with frame reordering are rejected when decode timestamps are unavailable; other readable cameras are retained. When part of an episode is damaged or unreadable, such as its state, a signal, its metadata, a sidecar or one camera, that part is left out and named on the episode, and the episode keeps every camera that can still be read. Each episode is read in a separate process, so a file that hangs or crashes a native reader is read again with its cameras alone instead of stopping the upload. A camera whose own timestamps disagree with the declared frame rate is placed by its timestamps, and the disagreement is flagged. A video used as it was uploaded is decoded once while it is prepared, so frames the decoder hides after damage are named with their span. A damaged ZIP is read through both of its indexes, and a member whose two names disagree is read as an upload of its own, so no table, task or metadata of the upload it came from describes it. Archives inside archives are opened at any depth within the upload's file and byte limits, and a member the filesystem cannot hold is left out and named while the rest of the upload is read.

For example, download and prepare the single MolmoAct2 episode listed in the quickstart:

```console
uv run python -m prepare molmo prepare --episodes configs/quickstart/teleop.txt \
  --out data/episodes/molmo/quickstart
```

The adapter retrieves the source files needed for that episode, including shared video packs, and preserves its camera offsets and state/action trajectories. Preparation makes no model calls.

For a custom board, create [`data/boards/mine/manifest.json`](configs/quickstart/board.json) with the dataset, prepared episodes and run paths. [configs/quickstart/board.json](configs/quickstart/board.json) shows the structure with published-dataset severity rules. Use [`python -m board build`](board/build.py), [`clips`](board/clips.py), [`serve`](board/serve.py), and [`static media`](board/static.py)/[`static site`](board/static.py); each takes [`--help`](board/__main__.py). Refresh the board after changing field definitions. To edit them in the local viewer, add [`--dictionary-job JOB`](board/serve.py) to [`board serve`](board/serve.py). The board answers only to `localhost` and, when [`--host`](board/serve.py) is a network address, to IP addresses and the machine's own names, so a reverse proxy that keeps the visitor's Host header, such as `tailscale serve` or Caddy, gets 403 until you pass that name with [`--allow-host NAME`](board/serve.py).

Export the same review job as a static viewer by preparing the media first, then building the site:

```console
uv run python -m board static media --board data/review/mine --clips data/review/mine/clips
uv run python -m board static site --board data/review/mine --clips data/review/mine/clips
```

The site is written to [`data/review/mine/static/<build_id>/`](board/static.py), with shared media in [`data/review/mine/static/media/`](board/static.py).

Paid staged runs require a clean tracked checkout. [`--resume RUN --why "reason"`](label/run.py) retains the run and remaining budget, using the same commit, dataset, prepared episodes and run kind; add [`--cap`](label/run.py) to raise the run's cap, for example to finish a run its cap stopped. Repeat [`--dataset`](label/run.py), [`--episodes`](label/run.py) and [`--kind`](label/run.py); pass the original harness options after [`--`](label/run.py). Cross-host writers require working shared filesystem locks.

[`python -m label.reparse RUN`](label/reparse.py) retries parsing stored replies that did not parse and rebinds their evidence without another call. Received but unusable replies, including truncated output, are never retried automatically. Add [`--retry-received EPISODE`](review/__main__.py) to the original review command to authorize another paid attempt for that episode, optionally raising [`--max-tokens`](review/__main__.py). For a staged resume, put these flags after [`--`](label/run.py); repeat the retry flag for additional episodes. Healthy outputs and prior billing receipts are retained. A request the provider refused with an error status, or one that never left the machine, cost nothing, so it is retried a few times and a plain resume sends it again. A request whose outcome is unknown, such as one that timed out after it was sent, may have been billed. It keeps its reservation, and only `--retry-received` for that episode sends it again, with the possible charge still counted against the cap. [`--force`](label/harness.py), also available in `review`, refreshes every selected episode, including completed ones. A cap refusal returns a nonzero exit status and retains each unfinished unit for resume.

Capped runs use verified model rates or a saved OpenRouter pricing quote and refuse unknown prices. Each request reserves an upper bound computed from the request itself, its text and images plus its full output limit, until the bill arrives. With the default 64,000-token output limit, the default model reserves about $3.50 per request, so a smaller [`--max-tokens`](review/__main__.py) reserves less. When the cap cannot cover a single request, `review` stops before any paid call and says so. Receipts preserve prior responses, charges and outstanding reservations across failures. [`board grids`](board/grids.py) reconstructs base request grids only when original input hashes can be verified; detail/sensor images are counted but not reconstructed there.

[`checks.timebase scan`](checks/timebase.py) detects MolmoAct2 episodes that play faster than real time. [`apply`](checks/timebase.py) adds its measurements and report flags to prepared episodes and existing labels without changing their clocks. Capture, source and sensor check failures are distinct from clean results. Optional hand pose uses [`board/hand_pose/modal_app.py`](board/hand_pose/modal_app.py) and [`board/hands.py`](board/hands.py); it is separate from ordinary annotation and has the licenses below.

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

Comparison caps apply per model run and gate caps per recording setup. Sampling may differ by model: compare saved requests before claiming identical inputs. [`--with-example`](compare/__main__.py) supplies a reference annotation; [`--routes`](compare/__main__.py) reuses resolution choices. [gate/selection.json](gate/selection.json) and [gate/cases.json](gate/cases.json) define the selected cases; [`gate score`](gate/score.py) evaluates the supplied runs. Its text checks test literal presence, not semantic correctness; ambiguous object claims require review and remain unresolved.

Comparisons distinguish missing responses from replies that failed to parse. Cost reports separate verified charges from unresolved reservations; gate cost rates exclude calls without a response or a verified invoice. Cold-cost estimates require the default model and a single paid attempt per unit.

[`uv run --frozen --offline pytest -q -ra`](.github/workflows/ci.yml) runs offline tests; viewer tests also require Node.js. CI runs the same command as four shards side by side, each a set of whole test files balanced by the times in [`tests/shard_times.json`](tests/shard_times.json); set [`ARGUS_TEST_SHARD=1/4`](tests/sharding.py) to run one shard locally. CI uses the locked environment, Node, ffmpeg and ffprobe without provider credentials or dataset access. Secret scans cover the committed tree and reachable history; they do not grant permission to publish uploaded data.

</details>

## Credits

The capture checks in [`checks/vendor/public_dataset_adapter_qc.py`](checks/vendor/public_dataset_adapter_qc.py) are Sambhav Gupta's, from public-dataset-adapter (commit 99d0a9e), with the changes listed in that file's header. The hand pose model is ACE-Ego-Hand (Yufei Liu et al., [arXiv:2608.20308](https://arxiv.org/abs/2608.20308)), built on Wan2.2, VideoX-Fun and MANO (Romero, Tzionas and Black, 2017).

## Licenses

The code is Apache-2.0 ([`LICENSE`](LICENSE)), and the few files that carry material under another license are listed in [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt). Full datasets and optional components are downloaded at run time; the architecture chart includes attributed image excerpts. The labels Pantheon publishes, on the data dashboard, in its downloads and in the release, are licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), so anyone may use them for any purpose with credit to Pantheon. The footage they describe keeps its dataset's license.

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
