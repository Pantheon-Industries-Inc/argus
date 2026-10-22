"""The instructions every episode of a rig shares, exactly as the model receives them.

A request has three parts: these instructions (identical for every episode of a rig with the same instruction
presence and the same recorded or video only variant, so the provider serves them from its prompt cache), then
the facts about one episode (label/episode.py: its dataset, cameras, sampled instants, recorded state and
instruction), then its frames. The instructions are, in order: what the footage
is, the output schema, why the episode is labelled and what counts as a problem (the data contract), how to use
the episode's instruction or the dataset's annotation (or, for a robot episode that has none, how to
name its tasks), how a robot recording is cut into its tasks and what each task is judged against (the episode's
instruction, the dataset's timed steps, or its own end state), and the note that keeps the timeline at the level of
action phases.

Three rigs: teleop_arms (a person teleoperates robot arms), handheld_gripper (a person does the task with
handheld grippers that carry cameras) and ego_head (a person wears a head camera and works with their own
hands). Wording that describes one rig's hardware, cameras, mistakes or success never appears in another's.
"""
from __future__ import annotations

import json
from pathlib import Path

FIXED_HEADER = ("HOW TO LABEL. These instructions are the same for every episode; the episode itself "
                "(its dataset, cameras, frames, recorded motion and instruction) follows after them.\n\n")

EVIDENCE_CONTRACT = (
    'This request has separate visual and sensor annotation contracts. Return one JSON object containing all '
    'required fields. Write the supplied sensor_findings and evidence_findings fields first, then the visual annotation fields. '
    'Sensor findings use their own source evidence, native timestamps and event granularity; the visual timeline '
    'and short key_events rules do not restrict them. A sensor-supported change can coincide with a visible action '
    'or occur when the hand or object is outside the RGB view. Preserve the difference between a measured change '
    'and an inferred physical explanation. adds_beyond_video means information beyond ordinary scene/task footage '
    'and the behavioral description. Sensor images, sensor video streams and measurement visualizations are '
    'eligible supporting evidence, even when rendered as RGB or included in the same frame grids. '
    'Empty findings remain valid.\n'
    '\nSENSOR MECHANICS ANALYSIS CONTRACT\n'
    'Analyze sensor mechanics as a primary task alongside visual annotation. Evaluate each applicable goal independently; '
    'a useful broad regional finding does not complete an applicable local spatial analysis.\n'
    'For supplied numeric responses, compare coherent trajectories across distinct source components, where supplied, '
    'with scene footage and available recorded or estimated configuration. Assess what response changes add beyond '
    'configuration changes. Unknown polarity still permits observed raw rises and falls. Infer relative loading only '
    'when the supplied source interpretation and convention support it, retaining their qualification and alternatives '
    'such as bending, changing fit or stale estimates.\n'
    'For supplied constituent arrays, separately examine within-region redistribution, changing spatial response patterns, '
    'and coherent localized sustained or pulsed responses beyond gross loading. Support local pattern claims with '
    'neighboring distinct source components across time; alternate IDs for the same component are not distinct evidence. '
    'Explain cancellation in a regional average when established by the supplied readings.\n'
    'When repeated manipulations are observed, compare corresponding phases using supplied sensor measurements, scene '
    'evidence and recorded configuration. Preserve useful agreement or contrast, including a quieter phase when its '
    'supported difference is meaningful, in the headline and detail with essential qualification visible. Put measured '
    'comparisons and source citations in observation and evidence. Do not infer unobserved continuity or unsupported mechanisms.\n'
    'Distinguish measured patterns from inferred contact movement or manipulation outcomes. Use only the supplied physical '
    'localization, response convention, calibration, reference and timing authority, preserving their stated limitations. '
    'An applicable analysis goal does not require a finding. Empty findings remain valid. When a supported comparison '
    'provides useful added information, preserve that contrast across headline and detail; do not bury it only '
    'in numeric observations. Use the existing output and citation contracts.\n')

EVIDENCE_REVIEW = (
    '\nApply the sensor and inspected-evidence contracts independently of visual milestone selection. Review the '
    'complete supplied trajectory through its final samples. Preserve distinct supported changes and their '
    'comparisons within a single visible action, rather than merging different manipulation implications into '
    'one broad phase. Do not infer an unseen physical action or outcome from a sensor change alone. Use exact '
    'supplied, verified episode-aligned measurement times for timed sensor findings, not a rounding to RGB sample '
    'times. Preserve all stated calibration, interpretation, coverage and alignment limits.\n')

# An episode with no recorded state and no other signal is video only (label/episode.py RECORDED_BLOCKS), and its
# shared instructions say nothing of a recorded motion it does not have. These are the words taken out, from the
# header, the robot rigs' tag list and the data contract; the 2026-10-02 audit found "the recorded motion" in the
# instructions of every video only robot episode. Tests check the resulting video-only capabilities and schema.
VIDEO_ONLY_HEADER = ("frames, recorded motion and instruction", "frames and instruction")
VIDEO_ONLY_TAG = ("state_video_mismatch, ", "")
VIDEO_ONLY_CONTRACT_WORDING = [
    (" from\n  the recorded motion)", ")"),
    ("; the recorded motion\n  disagreeing with the video)", ")"),
    ("(what the cameras show, what the recorded motion says, what the instruction\n  says, and how the task is done)",
     "(what the cameras show, what the instruction\n  says, and how the task is done)"),
    ("and breaks the link\n  between what is seen, what is recorded and what is asked",
     "and breaks the link\n  between what is seen and what is asked"),
]


def contribution_rules(r: str) -> str:
    """One contribution policy, with only the acting part's name changing between rigs."""
    actor = {"teleop_arms": "arm", "handheld_gripper": "gripper", "ego_head": "hand"}[r]
    return f"""- contribution is judged by a step's visible effect on the operator's evident current aim, using the
  whole available trajectory. Infer the aim from the actions and their context, not from an imagined task or
  an ideal route. Later actions are evidence of how a result was used, not categorical proof of success or failure.
    "advancing": achieves the current aim, makes useful intermediate progress, or effectively supports the work.
      Temporary results, preparation, repositioning, exploration and recovery can advance when the footage
      supports their useful role. A successful action stays advancing even if it is slow or indirect.
    "wasteful": an observed failed attempt, an unintended outcome, or a clearly unnecessary portion of motion.
      Establish what aim was unmet, what happened unintentionally, or what visible evidence makes that portion
      redundant. Motion toward an aim is not success when the attempted effect fails. A failed attempt stays
      wasteful even when the operator later recovers; judge the effective correction separately as advancing.
    "idle": the {actor} is not engaged in the task. Lack of object contact alone does not establish idle:
      necessary approach, repositioning, search or waiting can serve the task when the footage supports that role.
  Repeated failed attempts at the same aim are wasteful, but repetition alone is not evidence of failure.
  Intentional undo, disassembly, reset or a new activity does not retroactively make useful work wasteful.
  Assess each step's effect within the activity it served, whether or not the whole activity ultimately succeeds.
  Do not excuse an observed failure by inventing learning or exploration, or invent inefficiency from a
  hypothetical shorter route. If only part of an action is demonstrably unnecessary, identify that portion.
  Treat unseen continuation after a recording boundary or internal request cut as unknown, not proof that a result
  was unused or an attempt failed. When the evidence does not settle contribution, explain the uncertainty in notes.
"""

# The output schema of the robot rigs. Tags are offered for reuse, so one kind of problem gets one tag across
# episodes; they are never a closed list. Each timeline segment is a positional row, so its keys are not
# repeated hundreds of times.
SCHEMA = """
Produce a timeline of what happens, one segment per distinct action
phase (an approach, a grasp, a transport, a placement, a pour, a
release, a retreat). Do not collapse a multi-step activity into one
segment; TIMELINE GRANULARITY at the end of these instructions says
exactly how fine to cut.

Return ONLY JSON with this shape. Each timeline segment is one array whose values are in exactly the order of timeline_columns, so the keys are not repeated for every segment; everything else is ordinary JSON objects:
{
  "scene": {
    "objects": [
      {"attributes": ["<the visible features that show what it is: its shape, parts, markings>", "<color/size/state>"],
       "name": "<what it actually is, specific if you can read it>",
       "location": "<where in the workspace>"}
    ],
    "setting": "<one line describing the workspace>"
  },
  "timeline_columns": ["start_s", "end_s", "arm", "action", "object", "destination", "spatial_relation", "contribution", "progress", "notes"],
  "timeline": [
    [<start_s float>, <end_s float>, "left" | "right" | "both", "<free-form verb phrase for what the arm does>",
     "<object acted on, or null>", "<where it goes, or null>", "<e.g. 'into the bin', 'onto the plate', or null>",
     "advancing" | "wasteful" | "idle", <progress float 0..1>, "<notes, e.g. what makes this uncertain, or null>"]
  ],
  "task_summary": "<one imperative sentence describing the whole task>",
  "key_events": [
    {"t_s": <float>,
     "label": "<the task-critical milestone in plain words>",
     "kind": "<short tag, e.g. contact | subgoal_complete | phase>",
     "outcome": "success" | "failure" | "unclear",
     "note": "<why, only if failure or unclear>"}
  ],
  "state_changes": [
    {"t_s": <float>, "object": "<name>",
     "from": "<state before>", "to": "<state after>",
     "seen_in": "<which camera showed the state before and which showed it after, and the direction each was looking>"}
  ],
  "completion": {
    "task_completed": "success" | "success_then_undone" | "failure" | "partial" | "unclear",
    "success_predicate": "<the COMPLETE terminal state the given instruction asks for (with no instruction, the one task_summary describes): ALL sub-conditions that must hold for the task to be done, e.g. 'all hardware sorted into compartments AND the lid is closed AND both clasps are latched', not just the salient one, but do not pad it with trivia you cannot judge>",
    "completed_at_s": <float or null>,
    "goal_reached_at_s": <float or null>,
    "undone_at_s": <float or null>,
    "undone_by": "<what undid the reached goal, or null>",
    "reason": "<if not a clean success, why>"
  },
  "scene_graph": [
    {"t_s": <float>,
     "relations": ["<subject> <relation> <object>"]}
  ],
  "recovery": [
    {"failure_t_s": <float>, "failure": "<what went wrong>",
     "recovered": true | false,
     "recovered_at_s": <float or null>,
     "correction": "<what the operator ACTUALLY did to recover, or null if never>",
     "how_to_fix": "<the ideal corrective action>"}
  ],
  "instruction_variants": ["<paraphrase of the task>", "<another phrasing>"],
  "performance_review": "<1-2 sentences on overall execution quality: wasted motion, hesitation, regrasps, collisions, drops, near-misses>",
  "data_issues": [
    {"issue": "<what is anomalous, in plain words>",
     "category": "<short snake_case tag for the kind of data issue. Reuse one of these when it fits, so the same kind gets the same tag across episodes: <<ISSUE_TAGS>>. Coin your own when none fits.>",
     "severity": "low" | "medium" | "high",
     "t_s": <float when it starts, or null if it spans the episode>,
     "end_s": <float when it is over (the last frame it affects), or null if it spans the episode>,
     "evidence": "<which camera and time show it>"}
  ],
  "operator_mistakes": [
    {"issue": "<what went wrong in the performance, in plain words>",
     "category": "<short snake_case tag for the kind of mistake. Reuse one of these when it fits, so the same kind gets the same tag across episodes: <<MISTAKE_TAGS>>. Coin your own when none fits.>",
     "severity": "low" | "medium" | "high",
     "t_s": <float when the incident starts, or null if it spans the episode>,
     "end_s": <float when it is over: the aim is met, the object is back, or the operator moves on; null if it spans the episode>,
     "recovered": <true when the operator corrected it within the episode, false when not>,
     "deliberate": "yes" | "no" | "unclear",
     "evidence": "<which camera and time show it>"}
  ]
}

Rules:
- Name objects by what you actually see. There is no fixed vocabulary.
  If you cannot tell what something is, describe it by shape/colour.
- Ground visual annotations in the frames; when unsure, say so in notes rather than guessing.
  Separate sensor_findings can describe information supported by supplied sensor readings even when RGB does not
  show it. Keep the stated measurement limits.
- progress is judged with the whole available video in view, NEVER as an isolated snapshot.
  Ask what state the task had reached by this step and whether this step moves it toward the goal FROM THERE.
""" + contribution_rules("teleop_arms") + """
- A failure is an attempt that does not get what it was aimed at, or an outcome nobody aimed at: an object
  that unintentionally slips, falls, rolls or is knocked is a failure, even if it is recovered afterwards.
  An intentionally released, dropped or rolled object is judged against its evident aim. Each failed
  attempt is a key_event with outcome failure. A step succeeded only when the frames show its result (the
  object between the jaws in the frames after, the things meant to meet meeting, the object staying where
  it was put); when they do not show it, it is not a success, even when the instruction asks for that step.
- progress is per step, 0..1: how much of the whole task is done at that instant
  (the absolute level), 0 at the start, 1.0 only when the full success_predicate
  holds, a never-completed task plateaus below 1.0 at the best state reached, and
  it may dip if progress is undone (a placed object is knocked out). progress
  (absolute level) and contribution (direction given history) can diverge: while
  recovering from a fumble, contribution is advancing even though progress is only
  climbing back to where it had been.
- Times in start_s/end_s must be drawn from the frame timestamps you
  were given (interpolate between them if needed).
- Cover the whole episode with no large unexplained gaps.
- key_events is the SHORT list of moments a human would mark to judge
  progress on THIS task, at the granularity that matters for it, not the
  dense per-action list. Cup stacking -> each cup picked and each cup
  stacked. A card slide -> first contact and slide complete. Shirt
  folding -> each fold completed, not each grasp. Pick the natural unit
  of progress for the task in front of you. A short task with one such
  unit still gets its start and finish marked.
- A subgoal is a lasting step toward the goal: a change in the scene that the success_predicate depends
  on and that still holds once the hand or gripper that made it lets go, for example a cup on the stack,
  the drawer open, a fold made, a screw driven in. A pick, a grasp, a contact or a motion is not a subgoal,
  because what it gains is lost if the object is let go anywhere else. Every successful demonstration of a
  task passes through the same subgoals whatever the operator's style, and the goal's end-state (each
  task's, when there are tasks) is the last one, at the goal frame (completed_at_s, judged by its own rule
  below). Mark each subgoal as a key_event with kind subgoal_complete at the first moment its change holds
  once the hand or gripper that made it lets go, labelled with the state reached; nudging or straightening
  it afterwards neither moves it nor makes a new subgoal. Only the goal's end-state waits, as completed_at_s
  does, for the last adjustment: when the last piece is set down and then straightened, setting it down is a
  subgoal of its own at the first moment it holds, and the goal follows once the straightening is done.
  Subgoals do not lengthen the list: a pick, a placement that is later undone, a contact or a failed attempt
  that belongs on it keeps a kind of its own (a grasp, a placement), never subgoal_complete.
- completion judges the whole task on the BALANCE OF EVIDENCE, the way a fair
  human reviewer would, not as a strict verifier. Grade what the footage shows was
  ACHIEVED, not whether every detail was explicitly confirmed on camera.
  success: the core goal is clearly accomplished given the evidence; you do NOT
  need a frame that spells out every sub-condition, and a missing or unclear
  confirming view of an incidental detail (a latch you cannot see click shut, an
  occluded final state) is NOT evidence of failure and must not drag a clearly
  accomplished task down to partial. partial: a real, observable sub-goal is left
  undone. failure: the task is demonstrably not accomplished. unclear: the footage
  genuinely does not show enough to judge the CORE outcome.
  success_then_undone: the full success_predicate WAS clearly reached at some moment, and
  then something later in the episode undid it, so it does not hold at the end (the objects
  are reset or scattered again, knocked out of place, the result is taken apart, e.g. the
  recording kept running while the scene was reset for the next attempt). Use it instead of
  failure or partial whenever the goal was genuinely achieved and later lost: this is a
  demonstration that succeeded, and must not be confused with one that never got there. Only
  use it when the full predicate was actually seen to hold, not when the task merely came
  close.
- goal_reached_at_s is the FIRST frame at which the full success_predicate held, whether or
  not it then lasted: for success it equals completed_at_s; for success_then_undone it is the
  moment the goal was achieved (completed_at_s stays null, because the predicate does not hold
  to the end), with undone_at_s the first frame it no longer holds and undone_by what undid it;
  null for every other outcome.
- completed_at_s (the goal frame) is the FIRST frame at which the full
  success_predicate is satisfied AND then stays satisfied through the end of the
  episode. It marks the instant the world reaches its target end-state, i.e. the
  last task-completing action that puts the manipulated objects into their final
  required configuration (the release, the seating, the placement, the closure).
  It is NOT when the arms finish moving. Motions that come after the end-state is
  reached and do not change it, retracting, parking, returning the arms home,
  withdrawing the hands, never define or push back the goal frame, even when the
  instruction itself asks for such a retreat: a "pick it up, place it, then move
  away" task is complete at the place/release, not at the move-away, because
  moving away does not alter the objects' final state. The "stays satisfied to the
  end" test is how you reject false candidates: if a state briefly looks done but
  is then undone or is still being worked on (a clasp that pops back open, bolts
  placed while the lid is still open), the predicate was not truly held there, so
  the true goal frame is later. Example: bolts are all in their compartments at
  4:26 but the lid is still open and the clasps get fought for minutes after, so
  the predicate does not hold at 4:26; the goal frame is the instant the final
  clasp seats shut (the last action that completes the end-state), not the later
  moment the arms retract to park. null if the full predicate never holds and
  stays held. You see the whole video at once, so decide this globally.
- each key_event also gets an outcome, judged on the balance of evidence like
  completion above: did that specific subevent succeed (the cup landed, the fold
  held) or fail (dropped, missed, slipped). Use unclear only when the evidence
  genuinely does not show whether it happened; a missing close-up confirmation of
  an otherwise-evident outcome is not grounds for unclear.
- state_changes records physical results, not motions: an object going from
  open to closed, empty to full, unstacked to stacked. Only list real changes
  you can actually see. seen_in names the views the before and the after come
  from; a change read by comparing two views that look from different
  directions is not yet a change, so confirm it from comparable views first.
- performance_review is 1-2 plain sentences on how well the whole episode was
  executed: wasted motion, hesitation, regrasps, collisions, drops, near
  misses. Describe what you saw; do not invent problems that are not there.
- scene_graph gives the spatial relations between objects at the key moments
  (at least the start, each key event, and the end) as short "<a> <relation>
  <b>" phrases. Use whatever clear spatial or support relation you actually see
  (on, in, next-to, inside, holding, leaning-on, stacked-on, ...); there is no
  fixed set of relations. List only relations you can actually see.
- recovery pairs each failure with what the operator did about it. Repeated failed tries at one aim
  are one entry, from the first try (failure_t_s) to the one that works (recovered_at_s). failure_t_s and
  failure describe the failure (failure_t_s is when it happens, lining up with the wasteful
  attempt in the timeline and, for a milestone failure, a key_event with outcome
  failure). If the operator later corrects it within the episode, set recovered true,
  recovered_at_s to the time the correction succeeds, and correction to what they
  ACTUALLY did to recover. If it is never corrected, set recovered false,
  recovered_at_s null, correction null. how_to_fix is the ideal corrective action in
  either case. List only real failures; leave the list empty on a clean run.
- instruction_variants are 2-3 natural rephrasings of the task, the kind a
  person might actually say, so a policy is not tied to one wording.
"""


# The same schema for handheld grippers: the acting part is a gripper, and the end of a demonstration is a
# gripper lowered or set down rather than an arm parked.
HANDHELD_WORDING = [
    ("<free-form verb phrase for what the arm does>", "<free-form verb phrase for what the gripper does>"),
    ("It is NOT when the arms finish moving.", "It is NOT when the grippers finish moving."),
    ("retracting, parking, returning the arms home,\n  withdrawing the hands,",
     "lowering, setting down or pointing away a gripper,\n  withdrawing the hands,"),
    ("moment the arms retract to park.", "moment the grippers are lowered."),
]

# The list of tasks every robot request asks for, each judged against the most specific claim the dataset makes about
# its span (TASK_RULES): one of the dataset's timed steps, the episode's instruction, or its own end state.
TASK_FIELDS = """  "tasks": [
    {"start_s": <float>, "end_s": <float>,
     "task": "<what the demonstrator set out to do, as observed, as an imperative, e.g. 'stack the three red blocks'>",
     "objects": ["<the objects this task acts on>"],
     "judged_against": "subtask" | "instruction" | "own_end_state",
     "subtask": <the number of the dataset's timed step it is judged against, or null>,
     "success_predicate": "<the COMPLETE terminal state of what it is judged against (the timed step, the instruction, for one repeat of a cyclical instruction that repeat's, or the end state of what it evidently set out to do): ALL sub-conditions that must hold for it to be done, e.g. 'all hardware sorted into compartments AND the lid is closed AND both clasps are latched', not just the salient one, but do not pad it with trivia you cannot judge>",
     "outcome": "success" | "success_then_undone" | "failure" | "partial" | "unclear",
     "completed_at_s": <float or null>,
     "goal_reached_at_s": <float or null>,
     "undone_at_s": <float or null>,
     "undone_by": "<what undid the reached end state, or null>",
     "note": "<if not a clean success, why>"}
  ],"""

# A recording with an instruction: a task that stands in for the instruction by doing other work is judged against the
# instruction, and also against the end state of the work it did (own_outcome), so a competent demonstration of other
# work keeps an outcome of its own for relabelling (audit_2026-10-21 S1). Every other task leaves both null.
OWN_TASK_FIELDS = """     "own_success_predicate": "<for a task judged against the instruction whose work is not the work it asks for: the end state of the work it evidently set out to do; otherwise null>",
     "own_outcome": "success" | "success_then_undone" | "failure" | "partial" | "unclear" | null,
"""

# The robot schema of an episode with no instruction: the list of tasks in place of the one completion, with
# progress, the outcome and the goal frame judged per task. With an instruction the schema keeps its completion, word
# for word, and adds the list after it (schema).
TASKS_WORDING = [
    ('  "task_summary": "<one imperative sentence describing the whole task>",',
     '  "task_summary": "<one imperative sentence describing the recording\'s work: its one task, or its tasks in '
     'order>",'),
    ("""  "completion": {
    "task_completed": "success" | "success_then_undone" | "failure" | "partial" | "unclear",
    "success_predicate": "<the COMPLETE terminal state the given instruction asks for (with no instruction, the one task_summary describes): ALL sub-conditions that must hold for the task to be done, e.g. 'all hardware sorted into compartments AND the lid is closed AND both clasps are latched', not just the salient one, but do not pad it with trivia you cannot judge>",
    "completed_at_s": <float or null>,
    "goal_reached_at_s": <float or null>,
    "undone_at_s": <float or null>,
    "undone_by": "<what undid the reached goal, or null>",
    "reason": "<if not a clean success, why>"
  },""", TASK_FIELDS),
    ("""- progress is per step, 0..1: how much of the whole task is done at that instant
  (the absolute level), 0 at the start, 1.0 only when the full success_predicate
  holds, a never-completed task plateaus below 1.0 at the best state reached, and
  it may dip if progress is undone (a placed object is knocked out). progress""",
     """- progress is per step, 0..1: how much of the CURRENT task is done at that instant
  (the absolute level), 0 when the task starts, 1.0 only when its full success_predicate
  holds, a never-completed task plateaus below 1.0 at the best state reached, and
  it may dip if progress is undone (a placed object is knocked out). It resets toward 0
  when the next task starts; in a recording of one task it is the recording's progress. progress"""),
    ("""and the goal's end-state (each
  task's, when there are tasks) is the last one, at the goal frame""",
     """and each task's end-state is the
  last one, at the goal frame"""),
    ("""- completion judges the whole task on the BALANCE OF EVIDENCE, the way a fair
  human reviewer would, not as a strict verifier.""",
     """- each task's outcome is judged on the BALANCE OF EVIDENCE, the way a fair
  human reviewer would, not as a strict verifier."""),
    ("""  success_then_undone: the full success_predicate WAS clearly reached at some moment, and
  then something later in the episode undid it, so it does not hold at the end (the objects
  are reset or scattered again, knocked out of place, the result is taken apart, e.g. the
  recording kept running while the scene was reset for the next attempt). Use it instead of
  failure or partial whenever the goal was genuinely achieved and later lost: this is a
  demonstration that succeeded,""",
     """  success_then_undone: the task's full success_predicate WAS clearly reached at some moment,
  and then something undid it while it still had to hold (TASKS below says how long a task's
  result must hold): the objects are reset or scattered again, knocked out of place, the result
  is taken apart, e.g. the recording kept running while the scene was reset with no further task
  after it. Use it instead of
  failure or partial whenever the goal was genuinely achieved and later lost: this is a
  task that succeeded,"""),
    ("""(completed_at_s stays null, because the predicate does not hold
  to the end)""",
     """(completed_at_s stays null, because the predicate does not hold
  as long as it must)"""),
    ("""success_predicate is satisfied AND then stays satisfied through the end of the
  episode.""",
     """success_predicate is satisfied AND then stays satisfied for as long as the task's
  result must hold (TASKS below)."""),
    ("""The "stays satisfied to the
  end" test""",
     """The "stays satisfied" test"""),
    ("""judged on the balance of evidence like
  completion above""",
     """judged on the balance of evidence like
  the tasks' outcomes above"""),
    ("""- instruction_variants are 2-3 natural rephrasings of the task, the kind a""",
     """- instruction_variants are 2-3 natural rephrasings of the task that fills most of the
  recording, the kind a"""),
]

# Tags offered for reuse in the robot rigs' schema.
DATA_ISSUE_TAGS = ("instruction_mismatch, human_intervention, camera_fault, recording_fault, "
                   "state_video_mismatch, observability_loss, setup_change, scene_reset, idle_stretch, "
                   "truncated_episode")
OPERATOR_MISTAKE_TAGS = ("failed_grasp, dropped_object, knocked_object, collision, prolonged_struggle, "
                         "hesitation, unnecessary_motion, goal_undone, incomplete_task")


def camera_observation_block(views: list[tuple[str, str]]) -> str:
    """Optional pixel-grounded content descriptions using exact supplied view identities."""
    names = '; '.join(f'{view}: {alias}' for view, alias in views)
    return ('\nOBSERVED CAMERA CONTENT. The supplied view IDs and image-row labels are: ' + names + '. '
            'Include exactly one observation for every supplied view ID, including views whose content seems obvious. '
            'Aliases are not valid view IDs. Use unclear when the visible content is uncertain; do not omit that view. '
            'Add "camera_observations" to the final JSON, as a list of '
            '{"view": "<exact supplied view ID>", "content": "scene|sensor|unclear", '
            '"evidence": "<brief description of the visible content>"}. '
            'Use scene for a view showing the device, person, objects or workspace; sensor for a view of '
            'a sensing surface or sensor image rather than the surrounding work; unclear when the pixels '
            'do not establish either, or its content changes between these classes. Judge the visible content '
            'across the supplied footage, not its name, topic, slot or profile. '
            'These observations do not establish camera mounting, actor ownership, physical calibration '
            'or a recording fault. Preserve explicit human and source descriptions separately.\n')


def schema(r: str, recorded: bool = True, has_instruction: bool = True) -> str:
    """The output schema for a robot rig (the head camera has its own, EGO_SCHEMA). A video only episode is offered no
    tag for a state it does not have. An episode with an instruction keeps the completion that judges the recording
    against it, word for word, and lists its tasks after it; one with no instruction has only the tasks
    (TASKS_WORDING)."""
    tags = DATA_ISSUE_TAGS if recorded else _replace_once(DATA_ISSUE_TAGS, *VIDEO_ONLY_TAG)
    s = SCHEMA.replace("<<ISSUE_TAGS>>", tags).replace("<<MISTAKE_TAGS>>", OPERATOR_MISTAKE_TAGS)
    if has_instruction:
        s = _replace_once(s, '    "reason": "<if not a clean success, why>"\n  },\n',
                          '    "reason": "<if not a clean success, why>"\n  },\n'
                          + _replace_once(TASK_FIELDS, '     "note": "<if not a clean success, why>"}',
                                          OWN_TASK_FIELDS + '     "note": "<if not a clean success, why>"}') + '\n')
    else:
        for old, new in TASKS_WORDING:
            s = _replace_once(s, old, new)
    if r == "handheld_gripper":
        s = _replace_once(s, contribution_rules("teleop_arms"), contribution_rules(r))
        for old, new in HANDHELD_WORDING:
            s = _replace_once(s, old, new)
    return s


# The head camera's schema: activities as tasks[], and per step whether the hands are in view and what covers
# them. The two issue lists are the same as the robot rigs', with the tags a person's own work can have.
EGO_SCHEMA = """
Produce a timeline of what happens, one segment per distinct action
phase (an approach, a grasp, a transport, a placement, a pour, a
release, a retreat). Do not collapse a multi-step activity into one
segment; TIMELINE GRANULARITY at the end of these instructions says
exactly how fine to cut.

Return ONLY JSON with this shape. Each timeline segment is one array whose values are in exactly the order of timeline_columns, so the keys are not repeated for every segment; everything else is ordinary JSON objects:
{
  "scene": {
    "objects": [
      {"name": "<what it actually is, specific if you can read it>",
       "attributes": ["<color/size/state>"],
       "location": "<where in the workspace>"}
    ],
    "setting": "<one line describing the workspace>"
  },
  "timeline_columns": ["start_s", "end_s", "arm", "action", "object", "destination", "spatial_relation", "contribution", "progress", "hands_visible", "hands_wearing", "notes"],
  "timeline": [
    [<start_s float>, <end_s float>, "left" | "right" | "both", "<free-form verb phrase for what the hand does>",
     "<object acted on, or null>", "<where it goes, or null>", "<e.g. 'into the bin', 'onto the plate', or null>",
     "advancing" | "wasteful" | "idle", <progress float 0..1>, <hands_visible true | false>,
     "<what covers the hands if anything, e.g. 'blue nitrile gloves', 'oven mitt'; 'bare' if the bare skin is visible; null when hands are not visible>",
     "<notes, e.g. what makes this uncertain, or null>"]
  ],
  "task_summary": "<one sentence on the work the person does across the clip>",
  "viewpoint": "first_person | third_person - judge from the footage: first_person = a head/chest-mounted egocentric camera where you mostly see the wearer's own hands reaching out from the bottom of the frame and never their face; third_person = an external/tripod camera looking AT the person, where you can see their body, torso, or face. Do not assume; decide from what the camera actually shows.",
  "key_events": [
    {"t_s": <float>,
     "label": "<the task-critical milestone in plain words>",
     "kind": "<short tag, e.g. contact | subgoal_complete | phase>",
     "outcome": "success" | "failure" | "unclear",
     "note": "<why, only if failure or unclear>"}
  ],
  "state_changes": [
    {"t_s": <float>, "object": "<name>",
     "from": "<state before>", "to": "<state after>"}
  ],
  "tasks": [
    {"start_s": <float>, "end_s": <float>,
     "task": "<the unit of work the person set out to do, open-vocab, e.g. 'pour water into the kettle', 'chop the onion'>",
     "objects": ["<the objects this task acts on>"],
     "outcome": "success" | "partial" | "failure" | "unclear",
     "success_predicate": "<the end-state that means THIS unit of work is done>",
     "completed_at_s": <float or null>,
     "note": "<why, if partial/failure or unclear>"}
  ],
  "scene_graph": [
    {"t_s": <float>,
     "relations": ["<subject> <relation> <object>"]}
  ],
  "recovery": [
    {"failure_t_s": <float>, "failure": "<what went wrong>",
     "recovered": true | false,
     "recovered_at_s": <float or null>,
     "correction": "<what the operator ACTUALLY did to recover, or null if never>",
     "how_to_fix": "<the ideal corrective action>"}
  ],
  "instruction_variants": ["<paraphrase of the task>", "<another phrasing>"],
  "performance_review": "<1-2 sentences on how well the work was done: pace, rework, anything lost or dropped, time spent searching or waiting>",
  "data_issues": [
    {"issue": "<what is anomalous, in plain words>",
     "category": "<short snake_case tag for the kind of data issue. Reuse one of these when it fits, so the same kind gets the same tag across episodes: instruction_mismatch, other_person_same_object, camera_fault, recording_fault, observability_loss, setup_change, idle_stretch. Coin your own when none fits.>",
     "severity": "low" | "medium" | "high",
     "t_s": <float when it starts, or null if it spans the episode>,
     "end_s": <float when it is over (the last frame it affects), or null if it spans the episode>,
     "evidence": "<which camera and time show it>"}
  ],
  "operator_mistakes": [
    {"issue": "<what went wrong in the performance, in plain words>",
     "category": "<short snake_case tag for the kind of mistake. Reuse one of these when it fits, so the same kind gets the same tag across episodes: dropped_object, knocked_object, failed_grasp, rework, prolonged_struggle, hesitation, unnecessary_motion, incomplete_task. Coin your own when none fits.>",
     "severity": "low" | "medium" | "high",
     "t_s": <float when the incident starts, or null if it spans the episode>,
     "end_s": <float when it is over: the aim is met, the object is back, or the operator moves on; null if it spans the episode>,
     "recovered": <true when the operator corrected it within the episode, false when not>,
     "deliberate": "yes" | "no" | "unclear",
     "evidence": "<which camera and time show it>"}
  ]
}

Rules:
- Name objects by what you actually see. There is no fixed vocabulary.
  If you cannot tell what something is, describe it by shape/colour.
- Ground visual annotations in the frames; when unsure, say so in notes rather than guessing.
  Separate sensor_findings can describe information supported by supplied sensor readings even when RGB does not
  show it. Keep the stated measurement limits.
- progress is judged with the whole available video in view, NEVER as an isolated snapshot.
  Ask what state the task had reached by this step and whether this step moves it toward the goal FROM THERE.
""" + contribution_rules("ego_head") + """
- A failure is an attempt that does not get what it was aimed at, or an outcome nobody aimed at: an object
  that unintentionally slips, falls, rolls or is knocked is a failure, even if it is recovered afterwards.
  An intentionally released, dropped or rolled object is judged against its evident aim. Each failed
  attempt is a key_event with outcome failure. A step succeeded only when the frames show its result (the
  object held by the hand in the frames after, the things meant to meet meeting, the object staying where
  it was put); when they do not show it, it is not a success, even when the instruction asks for that step.
- progress is per step, 0..1, toward the CURRENT sub-task the operator is working on
  (see tasks below), NOT the whole session: 0 when they start a sub-task, 1.0 when
  that sub-task's end-state is reached, then it RESETS toward 0 when they move on to
  the next sub-task. It may dip if progress within a sub-task is undone. progress
  (level within the current sub-task) and contribution (direction given history) can
  diverge: while recovering from a fumble, contribution is advancing even though
  progress is only climbing back to where it had been.
- Times in start_s/end_s must be drawn from the frame timestamps you
  were given (interpolate between them if needed).
- Cover the whole episode with no large unexplained gaps.
- key_events is the SHORT list of moments a human would mark to judge
  progress within each activity in tasks, at the granularity that matters for it, not the
  dense per-action list. Cup stacking -> each cup picked and each cup
  stacked. A card slide -> first contact and slide complete. Shirt
  folding -> each fold completed, not each grasp. Pick the natural unit
  of progress for the task in front of you. A short task with one such
  unit still gets its start and finish marked.
- A subgoal is a lasting step toward an activity's end-state: a change in the scene that its
  success_predicate depends on and that still holds once the hand that made it lets go, for example a cup on
  the stack, the drawer open, a fold made, a screw driven in. A pick, a grasp, a contact or a motion is not a
  subgoal, because what it gains is lost if the object is let go anywhere else. Every successful attempt at
  an activity passes through the same subgoals whatever the person's style, and each task's end-state is its
  last one, at that task's completed_at_s. Mark each subgoal as a key_event with kind subgoal_complete at the
  first moment its change holds once the hand that made it lets go, even while the other hand still holds
  the work, labelled with the state reached; nudging or straightening it afterwards neither moves it nor
  makes a new subgoal. Only a task's end-state waits, as its completed_at_s does, for the last adjustment:
  when the last piece is set down and then straightened, setting it down is a subgoal of its own at the
  first moment it holds, and the task's end follows once the straightening is done. Subgoals do not lengthen
  the list: a pick, a placement that is later undone, a contact or a failed attempt that belongs on it keeps
  a kind of its own (a grasp, a placement), never subgoal_complete.
- THIS USES THE HUMAN-HAND ANNOTATION PROFILE: describe the person's work with their own
  hands. Determine viewpoint and camera mounting from the supplied evidence. There is NO robot and usually NO
  single overall goal, especially in a longer clip; the person may do a sequence of
  distinct activities. "arm"/left/right means the person's LEFT or RIGHT HAND. Do not
  force one global success/failure or one goal frame when there are several activities.
  Segment the clip into the distinct activities in `tasks`. A short clip that is genuinely
  one task yields one task, with its own goal frame.
- hands_visible is a per-step ground-truth: set it FALSE whenever NEITHER of the
  person's hands is actually in the supplied frame during that step, and TRUE when
  at least one hand (or a clearly hand-held object in the grip) is in shot. In
  first-person video the head/gaze often points away from the hands, so the hands drop
  below, behind, or out of the frame while the person leans in, looks over a shelf,
  reaches low, or turns away; in those steps you are inferring the hand action from
  body pose and context, not seeing it, so hands_visible is false. Judge it from what
  is actually in the frame for that step, not from whether an action is happening. Be
  honest and precise: it drives a viewer cue that marks exactly when the annotation is
  inferred rather than observed. Default TRUE only when a hand is genuinely visible.
- hands_wearing: when a hand IS visible, say what covers it if anything - "blue nitrile
  gloves", "oven mitt", "work glove" - or "bare" when the bare skin is visible; null only when no hand is
  visible. Track it per step: it changes if the person puts on
  or removes gloves partway through, so report what is actually on the hands at that step.
- tasks is the list of distinct activities, in time order. Each has a start_s/end_s
  span, an open-vocab `task` describing what the person is doing (e.g. "pour water
  into the kettle", "chop the onion"), the `objects` involved, a `success_predicate`
  (the end-state that means THAT activity is done), an `outcome`, and `completed_at_s`
  (its goal frame). A task is a coherent unit of intent, COARSER than key_events
  (milestones WITHIN a task) and coarser than the dense timeline. Segment by intent
  shift: a new task begins when the person turns to a different objective.
- each task's outcome is judged on BALANCE OF EVIDENCE, the way a fair human reviewer
  would, per task independently: success = that mini-task's end-state is clearly
  reached; partial = a real observable part of it is left undone; failure = it is
  demonstrably not accomplished (dropped, abandoned, undone); unclear = the available footage does not
  establish the activity's core outcome. A missing confirming
  view of an incidental detail is NOT failure. An abandoned attempt the operator gives
  up on is a failure (or partial) for that task, and does not taint the others. An activity already
  under way at the first frame, or still under way at the last, is judged only on what the clip shows:
  when the clip ends before its outcome can be established, use unclear and explain the boundary in its note.
  An unseen continuation or an unresolved core result is not evidence of partial completion or failure.
  Preserve any observed failed attempts within that activity as failures; uncertainty about its eventual
  outcome does not erase a demonstrated fumble, missed attempt or unintended result.
- each task's completed_at_s (its goal frame) is the FIRST frame that task's
  success_predicate is satisfied AND stays satisfied for the rest of that task's span:
  the last action that puts THAT task's objects into their final state (the release,
  seat, placement, closure), NOT when the hands move away to the next task. null if that
  task's end-state is never reached (partial/failure). There is deliberately no single
  episode-level goal frame; each completed task carries its own.
- each key_event also gets an outcome, judged on the balance of evidence: did that
  specific subevent succeed (the cup landed, the fold held) or fail (dropped, missed,
  slipped). Use unclear only when the evidence genuinely does not show whether it
  happened; a missing close-up confirmation of an otherwise-evident outcome is not
  grounds for unclear.
- state_changes records physical results, not motions: an object going from
  open to closed, empty to full, unstacked to stacked. Only list real changes
  you can actually see.
- performance_review is 1-2 plain sentences on how well the work was done, judged as a
  skilled person doing this job would judge it: pace, rework, anything lost or dropped, time
  spent searching or waiting. Describe what you saw; do not invent problems that are not there.
- scene_graph gives the spatial relations between objects at the key moments
  (at least the start, each key event, and the end) as short "<a> <relation>
  <b>" phrases. Use whatever clear spatial or support relation you actually see
  (on, in, next-to, inside, holding, leaning-on, stacked-on, ...); there is no
  fixed set of relations. List only relations you can actually see.
- recovery pairs each failure with what the operator did about it. Repeated failed tries at one aim
  are one entry, from the first try (failure_t_s) to the one that works (recovered_at_s). failure_t_s and
  failure describe the failure (failure_t_s is when it happens, lining up with the wasteful
  attempt in the timeline and, for a milestone failure, a key_event with outcome
  failure). If the operator later corrects it within the episode, set recovered true,
  recovered_at_s to the time the correction succeeds, and correction to what they
  ACTUALLY did to recover. If it is never corrected, set recovered false,
  recovered_at_s null, correction null. how_to_fix is the ideal corrective action in
  either case. List only real failures; leave the list empty on a clean run.
- instruction_variants are 2-3 natural rephrasings of the annotated goal when one is given, otherwise of
  the activity that fills most of the clip, the kind a
  person might actually say, so a policy is not tied to one wording.
"""


def what_this_is(r: str) -> str:
    """What the footage is, first, so the schema is read with the right rig in mind."""
    if r == "ego_head":
        return ("- WHAT THIS EPISODE IS: human activity annotated with the person's hands as the acting parts;\n"
                "  the recording is human demonstration footage collected to train robots and world models. The\n"
                "  person's hands, arms and body can appear, and that is expected. The profile does not establish camera\n"
                "  mounting or viewpoint. Hands and objects can leave the frame during the work;\n"
                "  hands_visible records when that happens, and it is not a data issue unless the work becomes\n"
                "  unobservable for a substantial stretch. Other people nearby are part of the surroundings, and so\n"
                "  is their own work: coworkers passing, working at the next station, restocking or taking from a\n"
                "  shared supply, or clearing a shared area are how real workplaces look and teach a model nothing\n"
                "  wrong. Another person matters only when they actively manipulate the same object the person is\n"
                "  manipulating at that moment, because a model would credit their action to the person's hands; that\n"
                "  is a low-severity data issue unless it takes over the task. When the dataset cuts continuous\n"
                "  footage into fixed-length files, a file that starts or ends in the middle of an activity is how\n"
                "  the dataset is packaged, not a data issue.\n")
    if r == "teleop_arms":
        return ("- WHAT THIS EPISODE IS: robot-arm activity; the profile alone does not establish the control method.\n"
                "  The recording is a demonstration collected to train robots. In the workspace the scene should change only\n"
                "  through the robot, unless the episode's own description (after these instructions) says a person\n"
                "  takes part in the task, in which case their part is intended. The room\n"
                "  around the workspace (walls, furniture, people further away) is sometimes visible, for\n"
                "  example when a gripper camera points up; that is the surroundings, not a problem by itself.\n")
    return ("- WHAT THIS EPISODE IS: gripper activity annotated with grippers as the acting parts; the profile\n"
            "  alone does not establish human holding, the control method or camera mounting. The recording is\n"
            "  a demonstration collected to train robots. Gripper bodies and fingers, and people's hands, arms\n"
            "  and bodies, can appear. Establish participant and device ownership from the pixels and explicit\n"
            "  source claims. When that evidence establishes a handheld demonstration with two grippers, the\n"
            "  other gripper can appear even when only one gripper's footage was sent. A similar housing with\n"
            "  its own fingers and handle or strap, independent motion and an object held in its fingers are\n"
            "  useful cues, but resemblance alone does not establish ownership or exclude a task object. When\n"
            "  established as the rig's other gripper, its actions are part of the demonstration. This profile's\n"
            "  gripper-only execution contract treats the fingers as the acting part, subject to explicit source task or actor rules.\n"
            "  A task object moved by a bare hand, arm or the gripper's housing outside that contract is worth\n"
            "  reporting: its change was not made by the declared acting part. This is an annotation contract,\n"
            "  not an inferred limit of a physical robot. Anyone else touching the scene is reported as usual.\n"
            "  The room around the task is sometimes visible; that is the surroundings, not a problem by itself.\n")


_WHY = {
    "robot": (
        "(policies that imitate the demonstrator, and world models that predict what the cameras will see next from\n"
        "  the recorded motion)"),
    "ego_head": (
        "(policies and world models pretrained on human hand-object interaction, which learn from the video how\n"
        "  hands grasp, move and change objects and what the scene looks like next)"),
}

_INTENDED = {
    "robot": (
        "- INTENDED VERSUS UNINTENDED. An outcome the task calls for is never an issue, "
        "even when it hides things from\n"
        "  the cameras (an object inside a container, under a cover, or folded out of sight). The same loss of view\n"
        "  caused by something the task did not call for is an issue, and so is the scene changed by anyone or\n"
        "  anything other than the demonstration."),
    "ego_head": (
        "- INTENDED VERSUS UNINTENDED. An outcome the work calls for is never an issue, "
        "even when it hides things from\n"
        "  the camera (an object inside a container, under a cover, or folded out of sight), and neither is a hand or\n"
        "  a held object passing close in front of the lens. The same loss of view caused by something the work did\n"
        "  not call for is an issue, and so is another person manipulating the object the wearer is working on, as\n"
        "  described above."),
}

_VISIBILITY = {
    "teleop_arms": (
        "- VISIBILITY IS JUDGED ACROSS ALL CAMERAS TOGETHER. A camera mounted on a gripper points wherever that\n"
        "  gripper points, so objects enter it, leave it and get cut off at its edges all the time, including when\n"
        "  the gripper parks or retreats at the end; that is how such a camera works and never an issue. The camera\n"
        "  that is not mounted on a gripper, where there is one, is what keeps the task observable. An object is out\n"
        "  of view only when most of it (more than about 60%) is outside every camera at the same time, and the task\n"
        "  did not intend it. Anything less, such as an object cut off at the edge of one view while it is mostly\n"
        "  visible in another, is not an issue at all, not even a low one."),
    "handheld_gripper": (
        "- WHAT KEEPS THE TASK OBSERVABLE. Where every camera is carried on a gripper, each sees only where the\n"
        "  person points it, so objects leave and re-enter the views all the time; that is how the rig works.\n"
        "  Where a camera is not carried on a gripper, it is what keeps the task observable. What a model needs\n"
        "  is for the task state to be recoverable at the moments that matter (each grasp, each placement, the\n"
        "  end state) from the views at that moment or just before and after. Observability is lost only when\n"
        "  the task state at such a moment cannot be established from any view, and the task did not intend it."),
    "ego_head": (
        "- WHAT KEEPS THE WORK OBSERVABLE. Objects and hands can leave the frame during the work and return later.\n"
        "  The presentation profile alone does not establish camera mounting. The work is observable when\n"
        "  the frames show what the hands did to the objects, even if not at every instant. Observability is a data\n"
        "  issue only when the outcome of the work cannot be read from the footage for a substantial stretch."),
}

_MAP = {
    "robot": (
        "the recording itself (a camera frozen, black, corrupted, covered or swapped; cuts, jumps or\n"
        "  repeated stretches; darkness, exposure swings or blur bad enough to hide the task for more than a\n"
        "  moment, since auto-exposure adjusting and blur during fast moves are normal; the recorded motion\n"
        "  disagreeing with the video); the scene (a person reaching in or moving objects, except where the episode's\n"
        "  own description, given after these instructions, makes a person part of the setting or the task; objects\n"
        "  moving on their own, the setup changing mid-episode); the extent (the episode starting mid-task "
        "or cut off before the\n"
        "  task ends, a long idle stretch or a hand reset of the scene recorded after the task is done)"),
    "ego_head": (
        "the recording itself (the camera frozen, black, corrupted or covered; cuts, jumps or repeated\n"
        "  stretches; darkness, exposure swings or blur bad enough to hide the work for more than a moment, since\n"
        "  auto-exposure adjusting and blur during fast head turns are normal; the camera not where the dataset\n"
        "  says it is worn); the scene (another person manipulating the object the wearer is working on, as described\n"
        "  above); the extent (for a clip that the dataset presents as one task, starting after the task began or\n"
        "  stopping before it ends; for fixed-length windows of continuous work, the clip edges are never an extent\n"
        "  issue; a long stretch with no hand work, such as waiting on a machine or walking away, is worth knowing\n"
        "  for trimming)"),
}

_MISTAKES = {
    "teleop_arms": (
        "performance are operator mistakes, and they go in operator_mistakes under the shared contribution\n"
        "  rule's evidence of failure or wasted motion: failed grasps, repeated failed attempts at the same evident\n"
        "  aim, unintended drops, knock-overs or collisions, fumbling, detours or hesitation demonstrably serving\n"
        "  no aim, unintended loss of an achieved goal within the current activity, or an evident task given up\n"
        "  while the recording continues. Intentional later reset is not itself a mistake (a scene reset after\n"
        "  the task is a data issue, above)."),
    "handheld_gripper": (
        "performance are operator mistakes, and they go in operator_mistakes under the shared contribution\n"
        "  rule's evidence of failure or wasted motion: missed or slipped grasps, repeated failed attempts at the\n"
        "  same evident aim, unintended drops, knock-overs or collisions, fumbling, detours or hesitation\n"
        "  demonstrably serving no aim, unintended loss of an achieved goal within the current activity, or an\n"
        "  evident task given up while the recording continues. Intentional later reset is not itself a mistake\n"
        "  (a scene reset after the task is a data issue, above). A person with handheld grippers often waves\n"
        "  a gripper at the start so its tracking can initialise, tests the jaws, shifts their grip on a\n"
        "  handle, sets one gripper down while working with the other, and lowers the grippers or points them\n"
        "  away at the end; these uses alone are not mistakes, and observed failures or wasted motion still\n"
        "  follow the shared rule. A static view from a gripper that has been set down is not a frozen camera."),
    "ego_head": (
        "work are operator mistakes, and they go in operator_mistakes. Moving a part from one hand to the other,\n"
        "  re-gripping, putting a tool down and picking it up again, and taking something apart so it can be worked\n"
        "  on are normal uses. These uses alone are not mistakes; observed failures or wasted motion still follow\n"
        "  the shared rule. Switching to another job when the work calls for it is not giving up."),
}


_SLACK_EGO = """- OPERATOR MISTAKES ARE THE FAILURES AND WASTEFUL STRETCHES, GROUPED AND GRADED. Every failure and every
  wasteful stretch in the timeline belongs to exactly one operator mistake, recovered or not, and every
  operator mistake is made of them. One mistake is one incident: the failed tries at one aim and the motion
  between them, a drop and fetching the object back, or one stretch of motion that serves no aim. t_s and
  end_s span the whole incident. deliberate says whether the incident looks staged; it never removes an
  incident or lowers its severity. Setup or settling in the first and last few seconds, brief pauses to look
  with nothing in hand, and idle stretches alone are not mistakes; observed failures or demonstrably wasted
  motion within them still follow the shared contribution rule."""


# The shared contract in the head camera's words: a video and its annotation, where the robot rigs have cameras,
# recorded motion and an instruction.
EGO_CONTRACT_WORDING = [
    ("(what the cameras show, what the recorded motion says, what the instruction\n  says, and how the task is done)",
     "(what the video shows, what the annotation says, and how the work is done)"),
    ("and breaks the link\n  between what is seen, what is recorded and what is asked",
     "and breaks the link\n  between what is seen and what is annotated"),
    ("a faithful, correctly labelled record of a demonstration", "a faithful, correctly labelled record of the work"),
    ("how well was the demonstration performed", "how well was the work done"),
    ("the label (the\n  instruction contradicts the observed task, object or required order)",
     "the label (the\n  annotation contradicts the observed activity, object or required order)"),
    ("An operator who does a different task than the instruction has an instruction\n  mismatch (a data issue), not "
     "an unfinished task (an operator mistake), and the outcome is still judged\n  against the instruction.",
     "A person who does different work than the annotation describes has an instruction\n  mismatch, not "
     "unfinished work."),
    ("judged against what the instruction asks and what the\n  demonstrator evidently did.",
     "judged against what the annotation says and what the\n  person was evidently doing."),
]


def data_contract(r: str, recorded: bool = True) -> str:
    """Why the episode is labelled and what counts as a problem, in the words of this rig. The purpose, the two
    lists and the severity scale are shared; what the cameras are, what a mistake looks like and what normal
    slack is are the rig's own. A video only robot episode (recorded False) has the words about a recorded motion
    taken out (VIDEO_ONLY_CONTRACT_WORDING)."""
    robot = r != "ego_head"
    c = _DATA_CONTRACT_BASE
    c = _replace_once(c, "<<WHY>>", _WHY["robot" if robot else "ego_head"])
    c = _replace_once(c, "<<INTENDED>>", _INTENDED["robot" if robot else "ego_head"])
    c = _replace_once(c, "<<VISIBILITY>>", _VISIBILITY[r])
    c = _replace_once(c, "<<MAP>>", _MAP["robot" if robot else "ego_head"])
    c = _replace_once(c, "<<MISTAKES>>", _MISTAKES[r])
    if robot and not recorded:
        for old, new in VIDEO_ONLY_CONTRACT_WORDING:
            c = _replace_once(c, old, new)
    if not robot:
        slack = c[c.index("- OPERATOR MISTAKES ARE THE FAILURES"):c.index("- SEVERITY IS TRAINING IMPACT")]
        c = _replace_once(c, slack, _SLACK_EGO + "\n")
        for old, new in EGO_CONTRACT_WORDING:
            c = _replace_once(c, old, new)
        c = (c.replace("demonstrator", "person").replace("a demonstration", "a recording")
             .replace("demonstrations", "recordings"))
    return c


_DATA_CONTRACT_BASE = """- WHY YOU ARE LABELLING THIS. These labels decide what happens to this recording in a robot-learning
  training set: keep it, trim it, mask part of it, correct its instruction, or drop it. Models trained on it
  <<WHY>> learn whatever the recording shows, including its mistakes and its accidents. A problem
  is anything that would teach such a model something wrong or unintended, or that makes part of the
  recording useless or misleading for learning, judged against what the instruction asks and what the
  demonstrator evidently did. You already see everything that happens in these frames; for each thing you notice, the question is
  whether a careful ML researcher preparing this dataset would want to know about it before training on it.
<<INTENDED>>
<<VISIBILITY>>
- TWO KINDS OF PROBLEM, KEPT IN TWO LISTS. Ask two separate questions of every episode.
  First, is this recording a faithful, correctly labelled record of a demonstration? Where it is not, that is
  a data issue, and it goes in data_issues. The map (not a checklist; report whatever you see, including kinds
  not listed): <<MAP>>; the label (the
  instruction contradicts the observed task, object or required order); observability (the
  loss of evidence needed to establish the task state). Describe the visibility loss without attributing
  intent unless the footage or an explicit source claim establishes it. A data issue is fixed in the dataset (trim,
  mask, relabel or drop) and says nothing about how skilfully the task was done.
  A free-form description need not list every action. An omitted action alone is not a contradiction
  unless the source declares that description exhaustive. Keep the full observed activity in the timeline.
  This does not excuse incorrect claims or missing and erroneous subtasks within declared timed coverage.
  Second, taking the recording as faithful, how well was the demonstration performed? Mistakes in the
  <<MISTAKES>> An operator who does a different task than the instruction has an instruction
  mismatch (a data issue), not an unfinished task (an operator mistake), and the outcome is still judged
  against the instruction. An operator mistake is recorded correctly; the question it raises is
  whether a model should learn from it.
- OPERATOR MISTAKES ARE THE FAILURES AND WASTEFUL STRETCHES, GROUPED AND GRADED. Every failure and every
  wasteful stretch in the timeline belongs to exactly one operator mistake, recovered or not, and every
  operator mistake is made of them. One mistake is one incident: the failed tries at one aim and the motion
  between them, a drop and fetching the object back, or one stretch of motion that serves no aim. t_s and
  end_s span the whole incident. deliberate says whether the incident looks staged; it never removes an
  incident or lowers its severity. Setup or settling in the first and last few seconds, brief pauses to look
  with nothing in hand, and idle stretches alone are not mistakes; observed failures or demonstrably wasted
  motion within them still follow the shared contribution rule.
- SEVERITY IS TRAINING IMPACT, not how visible or dramatic something looks, in both lists. It comes from
  two things together: how much of the episode the problem touches, and how directly it corrupts what a
  model takes from the episode (what the cameras show, what the recorded motion says, what the instruction
  says, and how the task is done). high: the episode cannot be used as it is; blindly training on it would
  teach something wrong at its core. A problem that runs through the whole episode and breaks the link
  between what is seen, what is recorded and what is asked is high even when it is subtle to spot, because
  every frame inherits it. medium: a bounded part of the episode is wrong or wasted, and the rest is good
  once that part is trimmed, masked or corrected. low: worth knowing, but a model trained on the episode as
  it is would barely be affected. For operator mistakes the scale grades the incident by what to do with its stretch: low is a
  brief incident over within about two seconds that leaves the scene as it was (a grasp that misses and the
  next one holds); medium is an incident that changes the scene (a drop, a knock, a spill) or lasts longer
  than about three seconds, so its stretch should be trimmed or masked; high is an incident that dominates
  the episode, or leaves the task wrong or given up.
  Between these examples, grade the incident by its duration and consequences; a brief recovered
  fumble remains a mistake. Inclusion is independent of severity.
  A problem that is easy to fix
  once someone knows about it (trim the tail, mask a stretch, rewrite the instruction) is still an issue and
  keeps its severity; being fixable is exactly why it must be reported. Do not inflate: preferences about
  style that would not change what a model learns are not issues. Do not deflate: never argue
  a real problem away because some kind of training might tolerate it.
- Each entry in either list carries the concrete evidence (which camera, which time). Leave a list empty
  when nothing belongs in it; a clean, well performed episode with both lists empty is a normal, common
  result.
"""


def _replace_once(text: str, old: str, new: str) -> str:
    """Exact single replacement that FAILS if the shared wording changed (never a silent no-op)."""
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"shared prompt wording changed: expected 1 x {old!r}, found {n}")
    return text.replace(old, new)


EGO_ANNOTATION_RULES = """
ABOUT THE DATASET'S ANNOTATION (the annotation itself is given below with the episode, when the dataset has one).
The dataset ships its own annotation of the episode: an overall goal and, when it has them, timed subtasks. They
are claims to check, not the truth about the video. Label what you see independently, then:
- emit a top-level "goal_alignment" object comparing the annotated goal to the activities you observed:
  {"matches_given": true | false, "relation": "aligned" | "narrower" | "broader" | "different" | "unrelated",
   "note": "<one line: if the annotation and the footage diverge, why; empty if aligned>"}
  (narrower: the person did the annotated goal and meaningfully more; broader: only part of it happened;
  different: coherent work, but a different activity; unrelated: the footage does not match the annotation;
  matches_given says whether the footage shows the annotated activity at all, so it is false only when the
  relation is different or unrelated: spans that are late, a subtask missing or invented, or the wrong hand
  named are errors in the annotation, reported below, not a different activity);
- report as data issues (tag instruction_mismatch) the annotation errors a model trained on these labels would
  learn wrongly from: a goal that describes a different activity, a subtask whose label does not describe what
  happens in its span, a span far enough off to put a label on the wrong stretch, a subtask missing from the
  annotation or invented by it, a subtask marked successful that visibly failed (or the reverse).
When the annotation gives a goal, the clip is a task episode: its tasks are the steps of that goal, and success is
the goal's end-state. When it gives timed subtasks but no goal, goal_alignment compares the subtask sequence as a
whole with what you observed. When it gives neither, each activity is judged on its own end-state.
"""


# How to grade against the episode's instruction and fill goal_alignment (the robot rigs).
INSTRUCTION_RULES = """
ABOUT THE EPISODE'S INSTRUCTION (the instruction itself is given below with the episode).
This instruction is metadata whose correctness is NOT guaranteed; treat it as a claim to check,
not as the truth about the video. The label uses it in two different ways, for two different
purposes. The timeline describes the trajectory, so it follows what the operator is ACTUALLY,
demonstrably doing (your independently-inferred task_summary), not the given goal. The outcome is
what a policy learns the instruction means, so it follows the instruction:
- contribution follows the shared rule in the schema, against the operator's evident activity,
  never simply against whether the step serves the given instruction.
- progress, like the outcome, is measured against the given goal: how much of what was ASKED
  holds at that instant. It reaches 1.0 only when success_predicate holds, so a demo that never
  reaches the given goal never reaches 1.0, however well it does its own task.
- CRITICAL: if the operator is clearly performing a different-but-coherent task than the given
  instruction, apply the same contribution rule and record
  the divergence in goal_alignment
  and as an instruction_mismatch data issue. A competent demonstration is good data even when it
  is mislabeled; the dense timeline must stay a faithful description of the real behavior.
- An instruction can say HOW as well as WHAT: a direction, a hand, a grip, the order of steps. A
  demonstration that reaches the end state the instruction asks for in another way did the given
  task, not a different one: judge the outcome and progress on that end state, choose the relation
  by what was done (aligned, or narrower or broader), and record the difference in manner as an
  instruction_mismatch whose severity is its training impact. A detail that changes the end state
  itself (which face of a block is up, which object ends where) is part of the end state, not manner.
  A step whose result already holds at the first frame (a cap already off, a plug already in) is
  part of the end state that holds, not part of the goal that never happened: keep the relation
  aligned. Absence of that motion alone is not an instruction_mismatch unless the source explicitly asserts
  its occurrence within this recording or declared timed coverage. Describe the missing observation as a
  coverage limitation, without inferring from the initial state or missing motion that the action occurred
  before recording. An initially satisfied state does not establish which earlier actions produced it.
  Still report observed task, method or order violations and missing required end states.
- Judge quantity and completion from the instruction's actual constraints. When its goal is satisfied
  and the operator repeats it or performs compatible additional activity, relation narrower can describe
  that added scope. Extra activity that preserves the requested result and violates no explicit constraint
  does not alone establish an instruction_mismatch. Report a mismatch when a violated constraint or a
  materially misleading instruction would teach the wrong activity, method, object or outcome. This does
  not exempt genuine fumbles or failed attempts from operator_mistakes.
- The objects the instruction names are part of its claim. One instruction is often written once
  for many recordings, so the object handled in this one can be a different kind of object from the
  one it names. Identify every object from what the frames show of it (its shape and parts, how it
  bends, folds or opens, any lettering or markings), as you would if the instruction named no
  object, and call it that in every field. Then compare it with the instruction. A handled object
  of a different kind from the one named is part of the end state (which object ends where), and it
  is an instruction_mismatch; a more specific or differently worded name for the same kind of
  object (a mug for a cup) is not.
Use the given goal for the compliance question: completion.task_completed and
success_predicate are the terminal state of THIS GIVEN goal (did the demo satisfy what was
ASKED?), so a demo of a different task is a completion "failure" against the given goal even
though its own steps are advancing. Calling it success would teach a policy that the instruction
means that other task. Still fill task_summary with what you INDEPENDENTLY observe. Record materially
contradicted instruction claims in data_issues; a narrower or broader scope alone is not such a contradiction.
This does not change completion: a genuinely unmet requested goal remains partial or failed when the
available recording establishes that outcome; unseen continuation is unknown.

Also emit a top-level "goal_alignment" object comparing the given goal to what you
independently observed, so a mismatch between the label and the footage is captured explicitly:
  "goal_alignment": {
    "matches_given": true | false,  // does the footage actually depict the given goal at all
    "relation": "aligned" | "narrower" | "broader" | "different" | "unrelated",
      // aligned: what you saw is exactly the given goal.
      // narrower: the operator did the given goal AND meaningfully more.
      // broader: only part of the given goal actually happened.
      // different: coherent purposeful work, but a different task than the goal.
      // unrelated: the footage does not match the instruction (likely mislabeled).
    "note": "<one line: if given and observed diverge, why; empty if aligned>"
  }
A "different"/"unrelated" relation means the operator went off-script or the episode is
mislabeled - both are exactly the cases downstream filtering needs to catch. Decide goal_alignment
before completion, and keep the two consistent: they describe the same footage. A demonstration
that reaches the given goal and then does more is "narrower". An outcome of
success or success_then_undone (the given goal was reached) cannot sit beside a "different" or
"unrelated" relation (the footage never shows the given goal).
"""


# The timeline at the level of action phases: every distinct action, grasp, release, error and recovery is still
# its own segment.
LEAN_NOTE = """

TIMELINE GRANULARITY:
- Segment the timeline at the level of distinct ACTION PHASES, not per-second or per-frame.
  Merge contiguous frames of ONE continuous action by ONE arm into a SINGLE segment: a whole
  reach is one segment, a whole transport is one segment, a whole placement is one segment. A
  typical episode is a few dozen segments, not hundreds. This coarseness is deliberate.
- You MUST still emit a separate segment for every distinct action, every grasp, every release,
  every placement, every error, and every recovery, and whenever the acting arm or the
  manipulated object changes. Coarser segmentation may NEVER drop a real event; it only merges
  frames that are the same ongoing action.
- Keep values terse: short action/object strings, and set "notes" to null unless it carries real
  information.
- Everything else (completion, goal_alignment, key_events, data_issues, recovery,
  state_changes, scene) is unchanged in content; only the timeline's granularity and verbosity
  change. Do not drop detail from those sections.
"""


def lean(r: str, has_instruction: bool = True) -> str:
    """The note that keeps the timeline at the level of action phases, with the acting part named as the rig
    names it, and the sections it leaves unchanged named as the request asks for them."""
    if r == "handheld_gripper":
        s = _replace_once(LEAN_NOTE, "by ONE arm", "by ONE gripper").replace("acting arm", "acting gripper")
    elif r == "ego_head":
        s = _replace_once(LEAN_NOTE, "by ONE arm", "by ONE hand").replace("acting arm", "acting hand")
        return _replace_once(s, "(completion, goal_alignment, key_events, data_issues, recovery,\n  state_changes, "
                                "scene)",
                             "(tasks, goal_alignment, key_events, data_issues, operator_mistakes,\n  recovery, "
                             "state_changes, scene)")
    else:
        s = LEAN_NOTE
    return _replace_once(s, "(completion, goal_alignment,", "(completion, tasks, goal_alignment," if has_instruction
                         else "(tasks, goal_alignment,")


def fixed_instructions(r: str, *, has_instruction: bool = True, recorded: bool = True, has_steps: bool = False) -> str:
    """Everything before the episode's own facts. A robot request keeps the episode prompt when the episode has an
    instruction (the completion judges the recording against it, INSTRUCTION_RULES) and adds its list of tasks
    (TASK_RULES); with no instruction it asks only for tasks (NO_INSTRUCTION_RULES); when the dataset times its own
    steps, each task a step covers is judged against it (STEP_RULES). So a robot rig has up to eight variants
    (instruction present or not, timed steps or not, recorded or video only). The head camera keeps its own schema
    and annotation rules, with the rule for naming each task's end state when the clip has no annotated goal: four
    variants. Every episode of the same variant shares the cached prefix. recorded False (no recorded state and no
    other signal, label/episode.py is_recorded) takes out every word about a recorded motion. A saved task mode never
    changes the text."""
    head = FIXED_HEADER if recorded else _replace_once(FIXED_HEADER, *VIDEO_ONLY_HEADER)
    if r == "ego_head":
        return (head + what_this_is(r) + EGO_SCHEMA + data_contract(r) + EGO_ANNOTATION_RULES
                + ("" if has_instruction else no_instruction_rules(r)) + lean(r))
    body = head + what_this_is(r) + schema(r, recorded, has_instruction) + data_contract(r, recorded)
    if has_instruction:
        body += INSTRUCTION_RULES + task_rules(r, has_instruction=True, has_steps=has_steps)
    else:
        body += task_rules(r, has_instruction=False, has_steps=has_steps) + no_instruction_rules(r)
    return body + (step_rules(has_instruction) if has_steps else "") + lean(r, has_instruction)


def requested_outcome(r: str, *, has_instruction: bool, has_steps: bool = False) -> str:
    """Where the instructions of this request ask the reply to put its outcome, read from the text fixed_instructions
    sends for it: the one completion, judged against the instruction, when the request asks for it (it then lists the
    tasks too), otherwise the tasks, each with its own outcome."""
    text = fixed_instructions(r, has_instruction=has_instruction, has_steps=has_steps)
    if '"task_completed"' in text:
        return 'completion.task_completed'
    if '"tasks": [' not in text:
        raise RuntimeError('the instructions ask for no outcome field')
    return 'tasks'


# An episode with no instruction (a bare video, or a dataset that ships none) still needs a task to grade the
# outcome and progress against, and a description that any motion satisfies ("rearrange the objects") makes every
# episode a success and hides a goal that was reached and then undone.
NO_INSTRUCTION_RULES = """
ABOUT THE TASKS. This episode comes with no instruction, so each task that no timed step of the dataset covers (with
no steps, every task) is named and judged on its own end state. Name each as the most specific end state the
demonstrator evidently worked toward (the blocks in a row, a tower, the cloth folded in half, the cup in the bin),
read from what their actions were evidently trying to achieve and not from how far they got, never as a description
that any motion would satisfy (moving, repositioning or rearranging the objects). Its success_predicate states that
end state, and whatever the work was evidently meant to cover but did not is a shortfall in the outcome, not something
outside the task. When that end state was reached and then
lost before the recording moved on to other work, the outcome is success_then_undone. When the actions fit no single
end state, name the one that fits best and say in the task's note what leaves it uncertain.
"""


def no_instruction_rules(r: str) -> str:
    """NO_INSTRUCTION_RULES in the rig's words. A head-camera clip with no annotated goal gets the rule for naming
    each task's end state ("the food in the container gathered onto the fork", never the work stopping); its own
    schema still decides how the clip is cut into activities and which outcomes a task can have."""
    if r != "ego_head":
        return NO_INSTRUCTION_RULES
    return ("""
ABOUT THE TASKS. This clip comes with no annotated goal, so the end state of each of its tasks is yours to infer. Name
each as the most specific end state the person evidently worked toward (the blocks in a row, a tower, the cloth folded
in half, the cup in the bin), read from what their actions were evidently trying to achieve and not from how far they
got, never as a description that any motion would satisfy (moving, repositioning or rearranging the objects, or the
work coming to a stop). Each task's success_predicate states that end state, and whatever the work was evidently meant
to cover but did not is a shortfall in the outcome, not something outside the task. When the actions fit no single end
state, name the one that fits best and
say in the task's note what leaves it uncertain.
""")


# A robot recording is a list of tasks, and the model decides how many there are. Each task is judged against the most
# specific claim the dataset makes about its span: a timed step that covers it, else the episode's instruction, else
# its own end state. With an instruction the completion still judges the recording as a whole against it, as the
# episode prompt always did, and the tasks never change it.
TASK_RULES = """
TASKS. <<LEAD>>
How many tasks there are is yours to decide from what the demonstrator does; a recording of one demonstration is one
task.
- A task is a coherent unit of intent, COARSER than key_events (milestones within a task) and coarser than the
  timeline. A new task begins when the demonstrator turns to a different objective, or starts the same unit of work
  again after its end-state was reached: in a cycle, where the objects are reset and the same work repeats, each
  repeat is its own task, judged on its own. These belong to the task they serve and are never tasks of their own:
  stages toward one end state (building a stack in two halves, or packing a box item by item, is one task with
  subgoals), failed attempts and their retries, recovery from a fumble, regrasps, setup before the work, and
  parking or retreat after it. Stretches with no evident objective (idling, looking around, tidying between tasks)
  belong to no task; tasks need not cover the whole recording.
- Name each task as the most specific end state of what it evidently set out to do, never as a description any
  motion satisfies (moving, repositioning or rearranging the objects).
- judged_against names the most specific claim the dataset makes about the task's span, and the task's
  success_predicate and outcome are judged against it:
  <<CLAIMS>>
- Each task's outcome is judged per task independently: an abandoned attempt is a failure (or partial) for that
  task and does not taint the others. A task already under way at the first frame, or still under way at the last,
  is judged only on what the recording shows; say in its note what is outside this view, and use unclear when the
  recording ends before its outcome can be established. Internal request cuts are not evidence of a failed activity
  or a defect in the original recording.
- How long a task's own result must hold: it is undone when it is lost before the recording moves on to other work
  (knocked, slipping, taken apart, or reset with no further task after it), and then the task is
  success_then_undone. Resetting or reusing its objects to start another task, or the next repeat of a cycle, is the
  start of that work, not an undo.<<HELD>>
"""

_CLAIMS = {
    (True, True): ('"subtask" when one of the dataset\'s timed steps covers it (its work falls within the step\'s span\n'
                   '  and is the work the step names; subtask is that step\'s number), else "instruction" when it does the\n'
                   '  work the instruction asks for or stands in for it, else "own_end_state", the task\'s own end\n'
                   '  state.'),
    (True, False): ('"instruction" for a task that does the work the instruction asks for or stands in for it, and\n'
                    '  "own_end_state" (the task\'s own end state) for every other task; subtask is null.'),
    (False, True): ('"subtask" when one of the dataset\'s timed steps covers it (its work falls within the step\'s span\n'
                    '  and is the work the step names; subtask is that step\'s number), else "own_end_state", the task\'s own\n'
                    '  end state.'),
    (False, False): '"own_end_state" for every task, the task\'s own end state; subtask is null.',
}


def task_rules(r: str, *, has_instruction: bool, has_steps: bool) -> str:
    """TASK_RULES for a robot request: the claims a task can be judged against are only those the episode has. With an
    instruction the tasks come beside the completion and never change it: a task's own result can be undone only
    before the recording moves on, while the completion keeps the episode prompt's rule that the goal must hold to the
    end, where a scene reset for the next attempt undoes it."""
    lead = ("Besides completion, list the recording's tasks in tasks, in time order." if has_instruction else
            "Segment the recording into its tasks and list them in tasks, in time order.")
    held = ("""
- Each task's outcome, completed_at_s, goal_reached_at_s, undone_at_s and undone_by follow the rules for completion
  above, applied to the task and to what it is judged against, except how long its result must hold. The tasks
  never change completion, which judges the whole recording against the instruction by its own rules: work the
  instruction does not ask for never raises it, different work never passes it, and its goal must hold through the
  end of the episode, so a scene reset for the next attempt still makes completion success_then_undone.
- A task judged against the instruction whose work is not the work it asks for (it stands in for it) is also judged
  against the end state of the work it evidently set out to do, as a task with no instruction would be:
  own_success_predicate states that end state and own_outcome says whether the footage shows it reached, by the same
  rules. They never change the task's outcome against the instruction, or completion. Every other task leaves both
  null.""" if
            has_instruction else """
- task_summary is one sentence: the one task, or the recording's tasks in order.""")
    return (TASK_RULES.replace("<<LEAD>>", lead).replace("<<CLAIMS>>", _CLAIMS[has_instruction, has_steps])
            .replace("<<HELD>>", held))


# The dataset's own timed steps on a robot recording (annotation_subtasks), each a short prompt for the work in its
# span: claims to check, and what a task they cover is judged against.
STEP_RULES = """
ABOUT THE DATASET'S TIMED STEPS (the steps themselves are given below with the episode, numbered).
The dataset times its own steps of the work, each with a short prompt. They are claims to check, not the truth
about the video. Label what you see independently, then:
- judge each task a step covers against that step (TASKS above): its success_predicate is the end state the step
  names, and its outcome says whether the footage shows it reached. A step marked unsuccessful is the dataset's
  claim, checked like any other.
- report as data issues (tag instruction_mismatch) the step errors a model trained on these labels would learn
  wrongly from: a step whose prompt does not describe what happens in its span, a span far enough off to put a
  prompt on the wrong stretch, a step missing from the dataset or invented by it, a step marked successful that
  visibly failed (or the reverse).
<<ALIGNMENT>>"""

_STEP_ALIGNMENT = {
    True: """- goal_alignment compares the instruction with what you observed, as set out under ABOUT THE EPISODE'S
  INSTRUCTION; the steps' errors are data issues, above.
""",
    False: """- emit a top-level "goal_alignment" object comparing the dataset's steps, as one sequence, with the tasks you
  observed:
  {"matches_given": true | false, "relation": "aligned" | "narrower" | "broader" | "different" | "unrelated",
   "note": "<one line: if the steps and the footage diverge, why; empty if aligned>"}
  (aligned: the tasks are the steps' work; narrower: the steps' work and meaningfully more; broader: only part of
  it happened; different: coherent work, but different work; unrelated: the footage does not match the steps;
  matches_given says whether the footage shows the steps' work at all, so it is false only when the relation is
  different or unrelated: a step that is late, missing or invented is an error in the steps, reported above, not
  different work).
""",
}


def step_rules(has_instruction: bool) -> str:
    """STEP_RULES; goal_alignment compares the instruction when there is one, else the steps as a sequence."""
    return STEP_RULES.replace("<<ALIGNMENT>>", _STEP_ALIGNMENT[has_instruction])


def example_block(r: str, example_dir: str | Path | None) -> str:
    """An in-context example, off by default (the model comparison's with-example runs): one complete annotation
    of a different episode of the same rig, given with its own context. It closes the shared instructions, so it
    is identical for every episode that gets the same shared instructions and cached with them."""
    if not example_dir:
        return ""
    f = Path(example_dir) / f"example_{r}.json"
    if not f.exists():
        return ""
    ex = json.loads(f.read_text())
    ctx = ex["context"]
    return ("\n\nAN EXAMPLE OF A COMPLETE ANNOTATION. Below is a full annotation, in the output format above, of a "
            f"different episode: {ctx['dataset']}, {ctx['rig_words']}, {ctx['length_s']:.0f} s long, with the "
            f"instruction \"{ctx['instruction']}\". It was made from that episode's own frames. It shows the "
            "density, the logic, the reasoning and the first principles expected of your annotation. It is not "
            "content to copy: your episode is a different recording, usually of a different task, and everything in "
            "your annotation must come from its own frames.\n" + json.dumps(ex["labels"], ensure_ascii=False) + "\n")
