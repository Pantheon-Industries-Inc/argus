"""How the reader places numeric rows on footage when their recorded instants do not settle it."""

ALIGNED_ROWS = "row per frame"       # equal row and frame counts, assumed one row per frame
ALIGNED_ASSUMED = "assumed start"    # separate clocks, assumed common start
COARSE_CLOCK = "coarse clock"       # shared stamps, assumed placement within each stamp interval

ALIGNED_CAMERA = "assumed camera clock"  # recorded values on frames whose presentation timing is assumed
