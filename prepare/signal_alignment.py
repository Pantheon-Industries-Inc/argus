"""How the reader places numeric rows on footage when their recorded instants do not settle it."""

ALIGNED_ROWS = "row per frame"       # equal row and frame counts, assumed one row per frame
ALIGNED_ASSUMED = "assumed start"    # separate clocks, assumed common start
COARSE_CLOCK = "coarse clock"       # shared stamps, assumed placement within each stamp interval

ALIGNED_CAMERA = "assumed camera clock"  # recorded values on frames whose presentation timing is assumed
RECORDER_ARRIVAL = "recorder arrival"
RECORDER_ARRIVAL_LIMITATION = ("Recorder arrival order does not establish physical event synchronization "
                              "or measured sensor-to-camera delay.")

# Consumers must name the actual placement, without inferring that separate clocks shared a start.
PLACEMENT_TEXT = {
    ALIGNED_ROWS: "placed one row per frame as an assumption",
    ALIGNED_ASSUMED: "placed from both starts",
    COARSE_CLOCK: "placed within each stamp interval as an assumption",
    ALIGNED_CAMERA: "placed on the assumed camera clock",
    RECORDER_ARRIVAL: "placed on the recorder arrival clock; physical synchronization is unverified",
}
UNKNOWN_PLACEMENT = "placed using an unspecified alignment assumption"
PLACEMENT_FIELDS = {
    ALIGNED_ROWS: "placed_row_per_frame",
    ALIGNED_ASSUMED: "placed_from_both_starts",
    COARSE_CLOCK: "placed_within_stamp_intervals",
    ALIGNED_CAMERA: "placed_on_assumed_camera_clock",
    RECORDER_ARRIVAL: "placed_on_recorder_arrival_clock",
}


def placement_text(aligned_by: str | None) -> str:
    """Describe a placement conservatively, including unknown future alignment kinds."""
    return PLACEMENT_TEXT.get(aligned_by, UNKNOWN_PLACEMENT) if aligned_by else ""


def recorder_arrival(context: dict) -> bool:
    """A recorder-log presentation has no independently verified physical event clock."""
    clock = (context.get('source') or {}).get('presentation_clock') or {}
    return clock.get('basis') == 'mcap_log_time' and clock.get('physical_synchronization_verified') is not True


# How strongly a value's time on the video is established, strongest first (video_timing):
RECORDED = "recorded"            # its own recorded instants, on a clock the cameras share, with no assumption
ARRIVAL = "recorder arrival"     # the recorder's arrival clock the cameras share; physical delay unmeasured
PLACED = "placed on frames"      # tied to the frames by an assumption within a shared reference: stamps within an
                                 # interval, the camera's assumed clock, or as many rows as the video has frames
UNTIMED = "untimed"              # nothing ties it to the video: placed from both starts, or an unknown placement
TIMING_GRADES = (RECORDED, ARRIVAL, PLACED, UNTIMED)
UNTIMED_WORDS = "its timing relative to the video is unknown, as it shares no clock with the cameras"


def video_timing(meta: dict | None, context: dict | None = None, anchor: str | None = None) -> dict:
    """{grade, placement, timed, verified}: the one rule for when a recorded value happened relative to the video,
    which every reader of a signal's time calls (the prompt, the checks, the evidence, the board and the export).
    meta is the signal's (or a contact's, or an evidence sensor's timing) entry with aligned_by, camera_aligned_by and
    clock_problem; context, the episode's, adds a recorder-arrival presentation, a state taken off the cameras and the
    anchor camera's own clock (anchor, the episode's anchor view, label/episode.py anchor, when the caller has it).

    timed says a value may be shown at a frame instant or a video time: its rows are tied to the camera frames by a
    clock they share or by the frames themselves. A placement from both starts (ALIGNED_ASSUMED) only assumes two
    recordings began together, and an unknown placement says nothing, so neither is timed: such a value is given
    only as a whole-recording summary. verified says its timing rests on no assumption at all (RECORDED), the bar a
    synchronized finding needs."""
    meta, context = meta or {}, context or {}
    by, camera, problem = meta.get("aligned_by"), meta.get("camera_aligned_by"), meta.get("clock_problem")
    if context.get("state_unaligned") or (by and by not in PLACEMENT_TEXT) or by == ALIGNED_ASSUMED:
        grade = UNTIMED
    elif (by in (ALIGNED_ROWS, COARSE_CLOCK, ALIGNED_CAMERA) or camera or problem or meta.get("method")
          or _anchor_clock_assumed(context, anchor)):
        # a timing derived by a stated method, or an anchor camera whose own capture clock is unusable, places the
        # values on the frames without a recorded instant
        grade = PLACED
    elif by == RECORDER_ARRIVAL or recorder_arrival(context):
        grade = ARRIVAL
    else:
        grade = RECORDED
    return {"grade": grade, "placement": by or (RECORDER_ARRIVAL if grade == ARRIVAL else None),
            "timed": grade != UNTIMED, "verified": grade == RECORDED}


def _anchor_clock_assumed(context: dict, anchor: str | None) -> bool:
    clocks = context.get("camera_clock") or {}
    if not clocks:
        return False
    if anchor is None:
        from label.episode import views
        anchor = views({"context": context})[0]
    return bool(clocks.get(anchor))


def timed(meta: dict | None, context: dict | None = None) -> bool:
    """Whether a value may be shown at a frame instant (video_timing)."""
    return video_timing(meta, context)["timed"]
