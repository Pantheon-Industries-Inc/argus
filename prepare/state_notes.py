"""Reasons a recording has no usable arm state. Shared by readers and request wording; state_note gives the cause."""

LAYOUT = "layout"
NOT_RECORDED = "not_recorded"
UNREADABLE = "unreadable"
SHORT = "short"
ASSUMED_CLOCK = "assumed_clock"

STATE_WHY = {
    LAYOUT: "a state is recorded, but not in a layout our checks read: its width, its value names, which arm is "
            "which, or rows that cannot be lined up with the frames",
    NOT_RECORDED: "the recording holds no state at all",
    UNREADABLE: "a file that holds the state, or may hold it, could not be read, or was damaged before any of its "
                "messages",
    SHORT: "an arm's state does not cover the footage: it starts late, stops early or stops inside it",
    ASSUMED_CLOCK: "a contributing arm or channel needs an assumption to place its readings on the footage; "
                   "the state note names the clock limitation",
}
