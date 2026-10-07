"""Read-only, CPU grasp measurements and an approval report; never builds data.

Run ``python -m so101.audit_grasps --dataset /path/to/pick_place_v1 --output audit``.
The global action extrema define the literal requested gripper thresholds. A
separate episode-span sensitivity report exposes their limits without silently
changing the primary measurements or approving exclusions.
"""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


FPS = 30
FORCED_BAD = frozenset((0, 6, 3, 51, 72))
METRICS = ("hover_s", "close_attempts", "dither", "close_duration_frames", "roll_alignment_deg")
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
USD_LIMITS_DEG = np.array(
    [(-110, 110), (-100, 100), (-100, 90), (-95, 95), (-160, 160), (-10, 100)],
    dtype=float,
)
# Freeze the recorder's calibration instead of importing its repository at runtime.
CALIBRATED_LIMITS_DEG = USD_LIMITS_DEG.copy()
CALIBRATED_LIMITS_DEG[0] = (-116.279296875, 119.794921875)
CALIBRATION_PROVENANCE = {
    "source": "Workshop/source/sim_to_real_so101/assets/real_setup.json:sim_joint_mapping",
    "source_sha256": "019cba78477c50151a473691d453549d84cd52b33e1e0508097bde5983fd1e0e",
    "conversion_source": "Workshop/source/sim_to_real_so101/utils/pick_place_eval.py",
    "conversion_source_sha256": "0bee65e1b79af4086f6d397e200f11dd27aca77963be9773f149883e273e0b51",
    "mapping_degrees": CALIBRATED_LIMITS_DEG.tolist(),
    "clamp_degrees": USD_LIMITS_DEG.tolist(),
}


@dataclass(frozen=True)
class Episode:
    index: int
    actions: np.ndarray
    joints_deg: np.ndarray
    sidecar: dict


def square_angle(degrees):
    """Signed angle folded into [-45,45), using the cube's 90-degree symmetry."""
    return (degrees + 45.0) % 90.0 - 45.0


def cube_face_yaw(quaternion):
    """Base-frame yaw of the cube axis with the largest horizontal projection."""
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise ValueError("Cube orientation must be a finite nonzero wxyz quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    axis = int(np.argmax(np.linalg.norm(rotation[:2], axis=0)))
    return float(np.degrees(np.arctan2(rotation[1, axis], rotation[0, axis])))


def calibrated_state_degrees(states):
    """Map recorded -100..100 arm / 0..100 gripper values to calibrated degrees."""
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] != 6 or not np.isfinite(states).all():
        raise ValueError("States must be a finite (frames,6) array")
    normalized = (states + 100) / 200
    normalized[:, 5] = states[:, 5] / 100
    low, high = CALIBRATED_LIMITS_DEG.T
    return np.clip(low + normalized * (high - low), *USD_LIMITS_DEG.T)


def episode_metrics(
    actions, joints_deg, sidecar, closed, opened, speed_threshold, fps=FPS, min_lift=0.01
):
    """Measure a literal downward global-midpoint crossing sustained for ten frames.

    Lift must be a new rise above the cube height at close, and must occur later.
    The initial-height diagnostic is separate and cannot prove a post-close lift.
    Missing values remain None; absence under a threshold is not physical failure.
    """
    actions, joints_deg = np.asarray(actions, dtype=float), np.asarray(joints_deg, dtype=float)
    n = len(actions)
    positions = np.asarray(sidecar["position"], dtype=float)
    quaternions = np.asarray(sidecar["orientation_wxyz"], dtype=float)
    if (
        n < 2
        or actions.shape != (n, 6)
        or joints_deg.shape != (n, 6)
        or positions.shape != (n, 3)
        or quaternions.shape != (n, 4)
        or not all(np.isfinite(a).all() for a in (actions, joints_deg, positions, quaternions))
        or np.any(np.linalg.norm(quaternions, axis=1) == 0)
    ):
        raise ValueError("Episode arrays must have matching, finite timelines")
    if (
        fps != FPS
        or not np.isfinite([closed, opened, speed_threshold, min_lift]).all()
        or closed >= opened
        or speed_threshold < 0
        or min_lift <= 0
    ):
        raise ValueError("Invalid 30 fps audit thresholds")
    g = (actions[:, 5] - closed) / (opened - closed)
    below = g < 0.5
    crossings = np.flatnonzero(below[1:] & ~below[:-1]) + 1
    sustained = [int(i) for i in crossings if i + 10 <= n and below[i : i + 10].all()]
    close = sustained[0] if sustained else None
    z = positions[:, 2]
    lifts = (
        np.flatnonzero((z - z[close] >= min_lift) & (np.arange(n) > close))
        if close is not None
        else []
    )
    lift = int(lifts[0]) if len(lifts) else None
    end = lift if lift is not None else n
    reopens = np.flatnonzero(below[:-1] & ~below[1:]) + 1
    attempts = sum(int(np.any((reopens > i) & (reopens < end))) for i in crossings if i < end)
    row = {
        "frames": n,
        "gripper_min": float(actions[:, 5].min()),
        "gripper_max": float(actions[:, 5].max()),
        "reaches_midpoint": bool(np.any(g >= 0.5)),
        "reaches_90pct_open": bool(np.any(g >= 0.9)),
        "cube_rise_anywhere_m": float(np.max(z - z[0])),
        "close_frame": close,
        "lift_frame": lift,
        "lift_ok": lift is not None,
        "close_attempts": attempts,
        "hover_s": None,
        "dither": None,
        "close_duration_frames": None,
        "roll_alignment_deg": None,
    }
    if close is None:
        return row
    velocity = np.diff(joints_deg[:, :5], axis=0) * fps
    row["hover_s"] = float(np.sum(np.linalg.norm(velocity[:close], axis=1) < speed_threshold) / fps)
    changes = velocity[max(0, close - 2 * fps) : close]
    row["dither"] = sum(
        int(np.sum(s[1:] != s[:-1]))
        for j in range(5)
        for s in [np.sign(changes[:, j][changes[:, j] != 0])]
    )
    starts = np.flatnonzero(g[:close] >= 0.9)
    if len(starts):
        start = int(starts[-1])
        finishes = np.flatnonzero(g[close:] <= 0.1) + close
        reopen90 = np.flatnonzero(g[close:] >= 0.9) + close
        if len(finishes) and (not len(reopen90) or finishes[0] < reopen90[0]):
            row["close_duration_frames"] = int(finishes[0] - start)
    yaw = cube_face_yaw(quaternions[close])
    row["roll_alignment_deg"] = abs(
        float(square_angle(joints_deg[close, 4] - square_angle(yaw - joints_deg[close, 0])))
    )
    return row


def classify(rows):
    """P75 linear percentiles screen hesitation; missing thresholds are provisional."""
    reference = [
        r
        for r in rows
        if r["episode"] not in FORCED_BAD and r["lift_ok"] and r["close_frame"] is not None
    ]
    if not reference:
        raise ValueError("No lifted, sustained-close reference episodes for percentiles")
    thresholds = {
        key: float(
            np.percentile([r[key] for r in reference if r[key] is not None], 75, method="linear")
        )
        for key in METRICS
        if any(r[key] is not None for r in reference)
    }
    for row in rows:
        bad, hesitant = [], []
        mandated = row["episode"] in FORCED_BAD
        if mandated:
            bad.append("mandated exclusion")
        if row["close_frame"] is None:
            bad.append("no sustained global-midpoint crossing (threshold unresolved)")
        if not row["lift_ok"]:
            bad.append("no measured post-close lift")
        hesitant.extend(
            f"{key}>P75"
            for key, threshold in thresholds.items()
            if row[key] is not None and row[key] > threshold
        )
        if (
            row["close_frame"] is not None
            and row["reaches_90pct_open"]
            and row["close_duration_frames"] is None
        ):
            hesitant.append("incomplete 90%-10% transition")
        row["classification"] = "bad" if bad else "hesitant" if hesitant else "clean"
        row["reasons"] = "; ".join(bad or hesitant)
        row["measurement_status"] = (
            "threshold_unresolved" if row["close_frame"] is None else "measured"
        )
        row["provisional_bad"] = bool(bad and not mandated)
        row["duration_measurable"] = row["close_duration_frames"] is not None
    return thresholds


def snapshot(root):
    """Streaming SHA-256 for every raw file, including videos and sidecars."""
    hashes = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[str(path.relative_to(root))] = digest.hexdigest()
    return hashes


def read_dataset(root, sidecars=None):
    """Validate LeRobot frames, metadata and cube timelines without writing them."""
    root = Path(root)
    sidecars = Path(sidecars) if sidecars is not None else root / "pick_place_meta"
    info = json.loads((root / "meta/info.json").read_text())
    if info["fps"] != FPS:
        raise ValueError("Audit requires 30 fps")
    expected_names = [j + ".pos" for j in JOINTS]
    for key in ("action", "observation.state"):
        if info["features"][key]["names"] != expected_names:
            raise ValueError(f"Unexpected joint ordering for {key}")
    count = info["total_episodes"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("Invalid total_episodes")
    frames = {}
    columns = ["episode_index", "frame_index", "timestamp", "action", "observation.state"]
    for path in sorted(root.glob("data/*/*.parquet")):
        for row in pq.read_table(path, columns=columns).to_pylist():
            key = (int(row["episode_index"]), int(row["frame_index"]))
            if key in frames:
                raise ValueError(f"Duplicate frame {key}")
            frames[key] = row
    episodes = []
    for ep in range(count):
        path = sidecars / f"episode_{ep:06d}.json"
        sidecar = json.loads(path.read_text())
        n = sidecar["num_frames"]
        if (
            sidecar["episode_index"] != ep
            or sidecar["fps"] != FPS
            or not isinstance(n, int)
            or isinstance(n, bool)
            or n < 2
            or sidecar["reference_frame"] != "robot base link (base)"
            or sidecar["units"] != "m"
        ):
            raise ValueError(f"Invalid sidecar {ep}")
        try:
            rows = [frames.pop((ep, i)) for i in range(n)]
        except KeyError as exc:
            raise ValueError(f"Missing dataset frame {exc.args[0]}") from exc
        if not np.allclose([r["timestamp"] for r in rows], np.arange(n) / FPS, atol=1e-5):
            raise ValueError(f"Invalid timestamps in episode {ep}")
        actions = np.asarray([r["action"] for r in rows], dtype=float)
        states = np.asarray([r["observation.state"] for r in rows], dtype=float)
        positions = np.asarray(sidecar["position"], dtype=float)
        quats = np.asarray(sidecar["orientation_wxyz"], dtype=float)
        if (
            actions.shape != (n, 6)
            or states.shape != (n, 6)
            or positions.shape != (n, 3)
            or quats.shape != (n, 4)
            or not all(np.isfinite(a).all() for a in (actions, states, positions, quats))
            or np.any(np.linalg.norm(quats, axis=1) == 0)
        ):
            raise ValueError(f"Invalid arrays in episode {ep}")
        episodes.append(Episode(ep, actions, calibrated_state_degrees(states), sidecar))
    if frames or sum(len(e.actions) for e in episodes) != info["total_frames"]:
        raise ValueError("Dataset totals disagree")
    if len(list(sidecars.glob("episode_*.json"))) != len(episodes):
        raise ValueError("Extra sidecars")
    return episodes


def measure_dataset(episodes, closed, opened, min_lift=0.01):
    """Primary global metrics plus explicitly non-authoritative local-span sensitivity."""
    speed = np.concatenate(
        [np.linalg.norm(np.diff(e.joints_deg[:, :5], axis=0) * FPS, axis=1) for e in episodes]
    )
    speed_threshold = float(np.percentile(speed, 25, method="linear"))
    rows = []
    for ep in episodes:
        row = {"episode": ep.index, "episode_index": ep.index}
        row.update(
            episode_metrics(
                ep.actions,
                ep.joints_deg,
                ep.sidecar,
                closed,
                opened,
                speed_threshold,
                min_lift=min_lift,
            )
        )
        low, high = float(ep.actions[:, 5].min()), float(ep.actions[:, 5].max())
        diagnostic = (
            episode_metrics(
                ep.actions, ep.joints_deg, ep.sidecar, low, high, speed_threshold, min_lift=min_lift
            )
            if low < high
            else {}
        )
        row.update(
            episode_span_midpoint=(low + high) / 2,
            episode_span_close_frame=diagnostic.get("close_frame"),
            episode_span_lift_frame=diagnostic.get("lift_frame"),
            episode_span_close_duration_frames=diagnostic.get("close_duration_frames"),
        )
        rows.append(row)
    return rows, speed_threshold, classify(rows)


def write_plots(rows, thresholds, closed, opened, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, key in zip(axes.flat, (*METRICS, "close_frame"), strict=True):
        vals = [r[key] for r in rows if r[key] is not None]
        if vals:
            ax.hist(vals, bins=min(15, max(1, len(set(vals)))))
        else:
            ax.text(0.5, 0.5, "Not measurable", transform=ax.transAxes, ha="center")
        ax.set_title(key)
        ax.set_ylabel("Episodes")
        if key in thresholds:
            ax.axvline(thresholds[key], color="red", label="reference P75")
            ax.legend()
    fig.tight_layout()
    fig.savefig(out / "distributions.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 5))
    xs = [r["episode"] for r in rows]
    ax.plot(xs, [r["gripper_max"] for r in rows], ".", label="episode maximum")
    ax.plot(
        xs,
        [r["episode_span_midpoint"] for r in rows],
        ".",
        label="episode-span midpoint (diagnostic)",
    )
    ax.axhline(closed + 0.5 * (opened - closed), color="red", label="global midpoint")
    ax.axhline(closed + 0.9 * (opened - closed), color="orange", label="global 90% open")
    ax.set_xlabel("Original episode index")
    ax.set_ylabel("Gripper action, recorded training units")
    ax.set_title("Sensitivity to global gripper extrema; diagnostic does not approve exclusions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "gripper_thresholds.png", dpi=150)
    plt.close(fig)


def make_summary(metadata):
    rows, thresholds = metadata["episodes"], metadata["thresholds"]
    global_gripper = metadata["global_gripper"]
    lines = [
        "# SO-101 grasp audit — approval required before data building",
        "",
        f"Source: `{metadata['source']}`. {len(rows)} episodes / {sum(r['frames'] for r in rows)} frames at 30 fps.",
        "",
        f"Global action closed/open: {global_gripper['closed']:.9g} / {global_gripper['opened']:.9g}; midpoint {global_gripper['midpoint']:.9g}; 90% open {global_gripper['open90']:.9g}.",
        f"Arm speed P25: {metadata['arm_speed_p25_deg_s']:.9g} degrees/s (five calibrated joint velocities, Euclidean norm).",
        "",
        "Close is the first downward midpoint crossing with at least ten consecutive below-midpoint frames; an initially closed gripper is not a crossing. Hover sums low-speed intervals from episode start to close, including settling before approach. Attempts count close-to-reopen cycles before the measured lift (or episode end if lift is absent); initial opening is excluded. Dither counts per-joint nonzero velocity sign reversals during the two seconds before close; exact zeros are ignored, with no smoothing. Duration spans the last sample at least 90% open before close to the first sample at most 10% open before returning to 90% open. Unobserved endpoints remain missing, never zero.",
        "",
        "Roll alignment is abs(fold90(wrist_roll - fold90(cube_face_yaw - shoulder_pan))) with the recorder's calibrated state conversion. Cube yaw uses the cube-local axis with the largest horizontal projection. This is a joint-angle proxy rather than gripper forward kinematics, so tilted cubes need video review. Lift requires a new base-Z rise of at least "
        f"{metadata['min_lift_m']:.4g} m above the height at close, strictly after close. This does not establish contact, release or task success. A rise above frame zero elsewhere is reported separately.",
        "",
        "P75 thresholds use NumPy's deterministic linear percentile on lifted, sustained-global-close episodes excluding the five mandated bad IDs. Any measured metric strictly above its P75 is hesitant; ties remain below the screen. An observed but incomplete 90%-10% transition is also hesitant. This upper-quartile screen finds demonstrations for review; percentile-only labels do not prove a failed grasp. A bad label comes from a mandated exclusion or a missing close/lift measurement; non-mandated bad labels remain provisional.",
        "",
        "Clean means no flag among measurable global metrics; it does not verify every metric. In particular, missing opening endpoints do not establish a fast closure. A percentile based on one measured duration equals that observation and provides no useful population estimate; review the opening-threshold policy before making closure-speed claims.",
        "",
        "| Metric | Measured N | Reference N | P0 | P25 | P50 | P75 | P95 | P100 | Hesitant threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    reference = set(metadata["threshold_reference_episodes"])
    for key in (*METRICS, "close_frame"):
        values = [r[key] for r in rows if r[key] is not None]
        ref_values = [r[key] for r in rows if r["episode"] in reference and r[key] is not None]
        quantiles = (
            np.percentile(values, [0, 25, 50, 75, 95, 100], method="linear") if values else []
        )
        formatted = [f"{v:.6g}" for v in quantiles] if values else ["—"] * 6
        threshold = f"{thresholds[key]:.9g}" if key in thresholds else "—"
        lines.append(
            f"| {key} | {len(values)} | {len(ref_values)} | "
            + " | ".join(formatted)
            + f" | {threshold} |"
        )
    for category in ("bad", "hesitant", "clean"):
        selected = [r for r in rows if r["classification"] == category]
        lines += [
            "",
            f"**{category.capitalize()}: {len(selected)} episodes / {sum(r['frames'] for r in selected)} frames**",
            "",
            ", ".join(str(r["episode"]) for r in selected) or "None",
        ]
    unresolved = [r["episode"] for r in rows if r["measurement_status"] == "threshold_unresolved"]
    provisional = [r["episode"] for r in rows if r["provisional_bad"]]
    sensitivity = metadata["sensitivity"]
    lines += [
        "",
        "**Threshold limitation and unresolved observations**",
        "",
        f"{len(unresolved)} episodes lack a sustained global-midpoint close; {sensitivity['never_reach_global_midpoint']} never reach the midpoint. Only {sensitivity['reach_global_90pct_open']} reach global 90% open, and duration is measurable in {sensitivity['global_duration_measured']}. Yet {sensitivity['cube_rise_anywhere']} episodes show a rise above their initial cube height somewhere. Missing global metrics cannot establish physical failure.",
        "",
        "Unresolved close IDs: " + (", ".join(map(str, unresolved)) or "None") + ".",
        "",
        "Non-mandated provisional bad IDs requiring review: "
        + (", ".join(map(str, provisional)) or "None")
        + ".",
        "",
        f"Episode-span min/max sensitivity finds {sensitivity['episode_span_close_measured']} sustained closes, {sensitivity['episode_span_post_close_lift']} later lifts, and {sensitivity['episode_span_duration_measured']} durations. These are diagnostic values in the CSV, not replacement thresholds. An episode's maximum need not be physically fully open, so adopting local spans also requires an explicit decision.",
        "",
        "**Proposed exclusions for approval: 0, 3, 6, 51, 72 (mandated).** Retain hesitant episodes in main pending approval; propose clean, measurable close/lift episodes for grasp windows. Review unresolved episodes and choose a gripper-threshold policy before treating additional bad labels as exclusions. No further exclusions are approved by this report.",
        "",
        "| Episode | Classification | Measurement status | Reasons |",
        "|---:|---|---|---|",
    ]
    lines.extend(
        f"| {r['episode']} | {r['classification']} | {r['measurement_status']} | {r['reasons'] or 'within measured thresholds'} |"
        for r in rows
    )
    lines += [
        "",
        "![Metric distributions](distributions.png)",
        "",
        "![Gripper threshold sensitivity](gripper_thresholds.png)",
        "",
        f"All {metadata['source_file_count']} raw file SHA-256 hashes matched before/after, including videos and sidecars. Source manifest aggregate: `{metadata['source_manifest_sha256']}`. No training data built. STOP: approve exclusions, hesitant handling and threshold policy before Phase 3.",
    ]
    return "\n".join(lines) + "\n"


def run_audit(root, out, sidecars=None, min_lift=0.01):
    root, out = Path(root).resolve(), Path(out).resolve()
    sidecars = Path(sidecars).resolve() if sidecars is not None else root / "pick_place_meta"
    if out.exists() or out == root or root in out.parents or out in root.parents:
        raise ValueError("Output must be a new directory separate from the source")
    if out == sidecars or sidecars in out.parents or out in sidecars.parents:
        raise ValueError("Output must be separate from the raw sidecars")
    before = snapshot(root)
    external_sidecars = not sidecars.is_relative_to(root)
    before_sidecars = snapshot(sidecars) if external_sidecars else {}
    episodes = read_dataset(root, sidecars)
    stats = json.loads((root / "meta/stats.json").read_text())["action"]
    closed, opened = float(stats["min"][5]), float(stats["max"][5])
    actual = np.concatenate([e.actions[:, 5] for e in episodes])
    if not np.isfinite([closed, opened]).all() or closed >= opened:
        raise ValueError("Invalid gripper action extrema")
    if not np.allclose([closed, opened], [actual.min(), actual.max()], atol=1e-5):
        raise ValueError("Metadata gripper extrema disagree with actual action values")
    rows, speed_threshold, thresholds = measure_dataset(episodes, closed, opened, min_lift)
    if before != snapshot(root) or (external_sidecars and before_sidecars != snapshot(sidecars)):
        raise RuntimeError("Raw source changed during audit")
    combined = {"dataset": before, "external_sidecars": before_sidecars}
    aggregate = hashlib.sha256(
        json.dumps(combined, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    reference = [
        r["episode"]
        for r in rows
        if r["episode"] not in FORCED_BAD and r["lift_ok"] and r["close_frame"] is not None
    ]
    metadata = {
        "schema_version": 1,
        "approval_required": True,
        "dataset_built": False,
        "source": str(root),
        "sidecars": str(sidecars),
        "source_manifest_sha256": aggregate,
        "source_file_count": len(before) + len(before_sidecars),
        "source_unchanged": True,
        "fps": FPS,
        "min_lift_m": min_lift,
        "lift_reference": "cube base-Z at close_frame; first later rise >= min_lift",
        "global_gripper": {
            "closed": closed,
            "opened": opened,
            "midpoint": (closed + opened) / 2,
            "open90": closed + 0.9 * (opened - closed),
            "open10": closed + 0.1 * (opened - closed),
        },
        "arm_speed_p25_deg_s": speed_threshold,
        "percentile_method": "linear",
        "hesitant_percentile": 75,
        "threshold_reference_episodes": reference,
        "thresholds": thresholds,
        "calibration": CALIBRATION_PROVENANCE,
        "forced_bad": sorted(FORCED_BAD),
        "categories": {
            key: [r["episode"] for r in rows if r["classification"] == key]
            for key in ("bad", "hesitant", "clean")
        },
        "proposed_exclusions": {
            str(ep): "mandated exclusion" for ep in sorted(FORCED_BAD) if ep < len(rows)
        },
        "proposed_grasp_window_episodes": [
            r["episode"]
            for r in rows
            if r["classification"] == "clean"
            and r["close_frame"] is not None
            and r["lift_frame"] is not None
        ],
        "hesitant_proposal": "retain in main; do not use for grasp_window unless explicitly approved",
        "non_mandated_bad_proposal": "review threshold ambiguity before exclusion or metric-policy change",
        "sensitivity": {
            "never_reach_global_midpoint": sum(not r["reaches_midpoint"] for r in rows),
            "reach_global_90pct_open": sum(r["reaches_90pct_open"] for r in rows),
            "global_duration_measured": sum(r["close_duration_frames"] is not None for r in rows),
            "cube_rise_anywhere": sum(r["cube_rise_anywhere_m"] >= min_lift for r in rows),
            "episode_span_close_measured": sum(
                r["episode_span_close_frame"] is not None for r in rows
            ),
            "episode_span_post_close_lift": sum(
                r["episode_span_lift_frame"] is not None for r in rows
            ),
            "episode_span_duration_measured": sum(
                r["episode_span_close_duration_frames"] is not None for r in rows
            ),
        },
        "episodes": rows,
    }
    out.mkdir(parents=True)
    with (out / "audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "source_sha256.json").write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (out / "summary.md").write_text(make_summary(metadata))
    write_plots(rows, thresholds, closed, opened, out)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/pick_place_v1"))
    parser.add_argument("--sidecars", type=Path, help="Defaults to dataset/pick_place_meta")
    parser.add_argument("--output", type=Path, default=Path("audit"))
    parser.add_argument("--min-lift", type=float, default=0.01)
    args = parser.parse_args()
    if not np.isfinite(args.min_lift) or args.min_lift <= 0:
        parser.error("--min-lift must be finite and positive")
    try:
        metadata = run_audit(args.dataset, args.output, args.sidecars, args.min_lift)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(f"Report: {args.output.resolve() / 'summary.md'}")
    print(f"Arm-speed P25: {metadata['arm_speed_p25_deg_s']:.9g} degrees/s")
    print("Hesitant thresholds (strict > reference P75; linear percentile):")
    for key, value in metadata["thresholds"].items():
        print(f"  {key}: {value:.9g}")
    for key, episodes in metadata["categories"].items():
        print(f"{key} ({len(episodes)}): {', '.join(map(str, episodes))}")
    print("Proposed exclusions: " + ", ".join(metadata["proposed_exclusions"]))
    print("Non-mandated bad labels are provisional; review global gripper-threshold ambiguity.")
    print("STOP: approve audit decisions before building datasets. Raw source hashes unchanged.")


if __name__ == "__main__":
    main()
