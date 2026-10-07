# SO-101 pick-place data

Source: original pick_place_v1 (75 episodes, 16448 frames, 30 fps), including pick_place_meta sidecars. The raw source is unmodified. No new demonstrations or domain randomisation.

Build command (from this repository after audit approval):

```bash
.venv/bin/python -m so101.build_dataset --source <raw-pick_place_v1> --audit audit/metadata.json --decisions audit/approved_decisions.json
# For sidecars outside the raw dataset, add --sidecars <pick_place_meta>.
```

Approved audit SHA-256: `67ebd212a155a3c4feb2bc762bdf906f3e2f24507e13aa812fb20b403244c531`.
Audited source-manifest SHA-256: `b52c0a9e67b7ebf88e647c15155ac103b7868c72d74792568d4d0144d7b4810c`.

| Subset | Episodes | Frames | Mixture weight |
|---|---:|---:|---:|
| main | 70 | 15250 | 1.0 |
| grasp_window | 18 | 2055 | 2.0 |

Main source episode IDs: 1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74.
Grasp-window source episode IDs: 7, 10, 13, 14, 15, 16, 19, 22, 24, 26, 29, 44, 48, 53, 54, 55, 58, 70.

Approved exclusions:

| Source episode | Reason |
|---:|---|
| 0 | Mandated bad episode exclusion approved by the user |
| 3 | Mandated bad episode exclusion approved by the user |
| 6 | Mandated bad episode exclusion approved by the user |
| 51 | Mandated bad episode exclusion approved by the user |
| 72 | Mandated bad episode exclusion approved by the user |

Audit thresholds (strictly greater than the measured reference P75 is hesitant; percentile method: linear):

| Metric | Threshold |
|---|---:|
| Arm speed P25, degrees/s | 4.68012632 |
| close_attempts | 0 |
| close_duration_frames | 12 |
| dither | 51 |
| hover_s | 0.958333333 |
| roll_alignment_deg | 34.4579565 |

Global gripper close/open and derived endpoints, in original recorded action units: `{"closed": 0.07704127579927444, "midpoint": 44.37596112862229, "open10": 8.93682524636388, "open90": 79.8150970108807, "opened": 88.67488098144531}`.
Minimum post-close cube lift: 0.01 m. Reference: cube base-Z at close_frame; first later rise >= min_lift.
Reference episodes for the percentile thresholds: 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 44, 48, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 71.

An episode is bad if mandated or if the primary close/lift measurement is absent; an observed incomplete 90%-10% closure is hesitant. Missing global-midpoint metrics can be threshold-unresolved; the approved episode decisions above govern inclusion. Roll alignment is the recorded calibrated wrist-roll/arm-direction proxy folded to 45 degrees, not a contact or task-success measurement.

Grasp windows contain approved clean episodes from close minus 1.5 s through the frame at lift plus 1.0 s, clipped to episode boundaries. Cameras: room/wrist; five arm joints plus gripper; absolute actions with horizon 16; original LeRobot arm units (-100 to 100) and gripper units (0 to 100), without conversion. Videos retain 640x480, 30 fps, AV1, yuv420p and no audio. Full episodes are byte-copied; clips use deterministic single-thread lossless AV1 and every decoded YUV and RGB frame is checked against the source.

Both subsets use GR00T v2.1 episode metadata, tasks, numeric statistics and the unchanged GR00T reader. These are ordinary Git files. The build enforces total size below 1 GB and every file below 95 MB; no Git LFS is needed. Counts, renumbering and clip bounds are also in build_report.json; full measurements in audit_metadata.json; source hashes in source_sha256.json; prepared-file hashes in prepared_sha256.json.

Encoder: ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023 the FFmpeg developers; libaom v3.8.2. Byte-identical builds within these encoder versions; other encoder versions may produce different AV1 bytes with the same verified YUV/RGB pixels.
