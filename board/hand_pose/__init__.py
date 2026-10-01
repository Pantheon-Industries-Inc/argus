"""The hand pose overlay for head-camera episodes: ACE-Ego-Hand keypoints computed on Modal and drawn on the dashboard.

modal_app.py runs the model on Modal and documents every command, core.py is the per-episode pipeline that runs
inside the Modal image, and specs.py writes each episode's clip and camera model. board/hands.py turns the output
into the overlay and the keypoint downloads.

The hand poses on the published board come from the second version of this pipeline. The first version drew some
hands hanging in space, away from any real hand, for two reasons.

- ACE-Ego-Hand's 2D head reads each joint as the attention-weighted mean of the image grid (a soft-argmax). When one
  hand slot attends to two hands at once, the wearer's and another person's or both of the wearer's, that mean is a
  point between them, on no hand. The pipeline now reads the 2D head a second time with only the attention near
  each slot's peak (core.mode_soft_argmax). core.combine_readouts keeps the model's own skeleton wherever the two
  readings agree, and where the model's reading is such a blend it draws the peak reading instead, scaled back to
  the hand's size. The peak reading is never used alone, because on its own it draws every hand at about half size.
- Gen-HumanEgo's fisheye was resampled to a virtual pinhole 110 degrees wide, which drew the wearer's hands small in
  the model's input. The pinhole is now 90 degrees wide and pitched down 30 degrees (specs.py).

A rerun also names its own output and specs volumes on Modal (HAND_POSE_OUT and HAND_POSE_SPECS, see modal_app.py),
so the second version was written beside the first and never over it.

We measured both versions on all 298 published head-camera episodes against an independent hand detector, MediaPipe
HandLandmarker, counting a keypoint as on a hand when it falls inside the detector's box around that hand grown by
15% of its larger side.

                                                                       first version   second version
  drawn keypoints on a detected hand, all frames                              69.8%            70.5%
  the same, on frames with three or more detected hands                       73.1%            82.0%
  drawn hands that sit between two detected hands, on those frames            17.6%             7.6%

Frames with three or more hands are where another person's hand is in view, which is where the blend happens. Per
episode, 130 improved, 20 got worse and 148 were unchanged (within 0.1 points of the first share). Egocentric-100K's
camera did not change, so its gain comes from the second readout alone. OpenAoE barely changes, because the blend
was already rare there (1.4% of drawn hands on frames with three or more detected hands).
"""
