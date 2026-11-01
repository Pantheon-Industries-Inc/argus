# Handoff: Argus labels in LeRobot

This branch holds only this file. It describes work to do, not work done. It was written on 2026-10-31 against Argus `main` at f78dfd0. Delete this file once the work lands.

## Goal

Make Argus labels load directly into LeRobot training, then get Argus named in LeRobot's official annotation docs. The point is recognition: every LeRobot user reads those docs, and Argus produces label types that LeRobot's own annotator does not.

There are two steps, in order. Step 1 lives in this repo and needs nobody's approval. Step 2 is an upstream contribution to `huggingface/lerobot`.

## What LeRobot has today (checked 2026-10-31)

Sources:
- Docs: https://huggingface.co/docs/lerobot/annotation_pipeline (source `docs/source/annotation_pipeline.mdx`)
- Schema: `src/lerobot/datasets/language.py`
- Pipeline: `src/lerobot/annotations/steerable_pipeline/` (`executor.py`, `validator.py`, `writer.py`, `modules/`, `prompts/`)
- CLI: `src/lerobot/scripts/lerobot_annotate.py`

How `lerobot-annotate` works:
- It serves one Qwen VLM (default `Qwen/Qwen3.6-27B`) through vLLM and drives it over the OpenAI-compatible API. It can dispatch itself to Hugging Face Jobs with `--job.target=h200`.
- Modules are `plan` (subtasks, plan, memory, task rephrasings), `interjections` (interjections and `say` tool calls) and `vqa`. The files are `modules/plan_subtasks_memory.py`, `modules/interjections_and_speech.py` and `modules/general_vqa.py`.
- Each module writes raw JSONL to `<root>/.annotate_staging/episode_{N:06d}/<module>.jsonl`. `StagingValidator` checks it, and one writer rewrites `data/chunk-*/file-*.parquet` in place. `--new_repo_id` with `--push_to_hub=true` publishes an annotated copy.
- Input is LeRobot v3.1 datasets only.

Schema, from `language.py`:
- There are two columns: `language_persistent` (a state that holds until replaced) and `language_events` (rows only on the frame whose timestamp matches).
- Persistent row fields are `role, content, style, timestamp, camera, tool_calls`. Event row fields are `role, content, style, camera, tool_calls`.
- `PERSISTENT_STYLES = {"subtask", "plan", "memory", "motion", "task_aug"}` and `EVENT_ONLY_STYLES = {"interjection", "vqa", "trace"}`.
- `EXTENDED_STYLES` is a set that projects append to at import time to register new styles. They must also add each new style to `PERSISTENT_STYLES` or `EVENT_ONLY_STYLES` to pick its column.
- `VIEW_DEPENDENT_STYLES = {"vqa", "trace"}` must carry `camera` set to an `observation.images.*` key, and every other style must have `camera=None` (`validate_camera_field`).

The validator requires all of the following:
- every event row lands exactly on a real frame timestamp, copied from the source parquet and never recomputed;
- subtasks form a gap-free cover, so every frame has exactly one active subtask;
- `plan` is refreshed at every interjection timestamp, and `memory` falls on subtask boundaries (a warning);
- every row goes to the column `column_for_style(style)` names.

The docs invite contributions: "contributions are very welcome", with examples like a new module for trajectory traces or affordances. Nothing in LeRobot produces outcomes, operator mistakes, recoveries, key events or object state changes. Those are the label types to bring.

LeRobot's `plan` module is converging on the design Midcentury and Macrodata use for subtasks: a describe-then-segment pass, an optional seeded relabel pass that sees the previous, current and next segment, and timestamped contact sheets at 2 FPS. Do not pitch Argus as better subtasks. Pitch it as the dense labels LeRobot cannot make.

## Step 1: an exporter in Argus (about 1 to 2 days)

Add a command, for example `python -m board export-lerobot`. It reads Argus's episode export (the same JSON the board downloads) and writes LeRobot's two columns into a LeRobot v3.1 dataset, or into a copy pushed to a new Hub repo.

Map the fields as follows:
- **Subtasks:** Argus actions and subgoals become `subtask` rows in `language_persistent`. Fill any gap in the Argus timeline so the cover is complete; choose the filler label deliberately and document it.
- **Dense findings:** each of these gets its own new style in `language_events`, registered through `EXTENDED_STYLES` and placed in `EVENT_ONLY_STYLES`. Proposed names are `outcome`, `mistake`, `recovery`, `key_event` and `state_change`. Keep `content` as plain text, with the structured detail as JSON inside it if needed, the way `vqa` does.
- **Timestamps:** snap every timestamp to the nearest real frame time read from the source parquet, and never compute it from a frame index divided by fps. Record the snap distance, and refuse or flag when it is larger than one frame.
- **Plan and memory:** Argus produces no `plan` or `memory` rows. Emit none rather than invent them. If the validator insists on them, find out which of its rules are errors and which are warnings before working around anything.
- **Validation:** run LeRobot's own `StagingValidator` (or the same checks) on the output in a test, so a dataset that passes is known to load into LeRobot recipes.

Scope limit: this works only for recordings that started as LeRobot datasets. MCAP, HDF5 and raw video have no LeRobot dataset to write into. State that in the README instead of converting formats.

Payoff once it works: push annotated copies of popular LeRobot datasets to the Hub. TU Munich already paid for Argus labels on DROID, ABC-130k, MolmoAct2, GM-100 and ABC-Sim (https://huggingface.co/datasets/nkalyanv/argus-robot-annotations, CC-BY-4.0), so those copies cost only upload time. Check each source dataset's license before republishing.

Timing: build this after the in-flight Argus work on the primary Mac has merged, because that work changes behavior and may change the export format the exporter reads.

## Step 2: the upstream LeRobot contribution (about 1 to 2 days, plus their review)

A full Argus module that calls a model will not fit upstream. LeRobot's modules all share its one vLLM client, and the maintainers will not take on a dependency on an outside harness that runs on frontier APIs. The contribution that fits their design has three parts:
- **Styles:** add `outcome`, `mistake`, `recovery`, `state_change` (and `key_event` if it holds up in step 1) to `language.py` with their columns, and document them in `docs/source/language_and_recipes.mdx`.
- **Importer:** an importer that loads outside annotations into `.annotate_staging/` so that the existing validator and writer do the rest. Argus is its first supported source, and the Argus side is step 1's mapping.
- **Docs:** a paragraph in `docs/source/annotation_pipeline.mdx` naming Argus as the source of those styles. That paragraph is the recognition.

Open an issue on `huggingface/lerobot` describing the styles and the importer before writing the PR. The main risk is a maintainer deciding the styles do not belong in the core list, and an issue settles that for an hour of work instead of two days.

## Related open PRs on Argus (raised from the second Mac, not merged)

- #167 adds CITATION.cff and a README Citation section with the blog's BibTeX (`li2026data2`, title "Argus: A Dense Annotator for Robotics Data", which is the title Calix set on the landing site on 2026-10-27).
- #168 sends OpenRouter app attribution headers (`HTTP-Referer`, `X-OpenRouter-Title`) from `_post` in `label/harness.py`, so Argus usage appears in OpenRouter's app rankings.

Both branch from f78dfd0. Rebase them onto `main` after the primary Mac's work lands, then merge. Data Review (data.pantheon.inc/review) needs a redeploy after #168 merges for its usage to count.
