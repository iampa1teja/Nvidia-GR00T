# Upstream provenance

Source: https://github.com/NVIDIA/Isaac-GR00T.git

Pinned commit: `51d4c89f72fda44cbf77285c6a8114b52676b8a1`. The tracked source working tree had no local differences before restructuring. NVIDIA's `gr00t/` sources are retained unchanged; project adaptations belong in `so101/` and runtime patches.

Official base: `nvidia/GR00T-N1.7-3B`, named by the pinned README, revision `2fc962b973bccdd5d8ce4f67cc63b264d6886495`.

Backbone: `nvidia/Cosmos-Reason2-2B`, revision `9ce19a195e423419c349abfc86fd07178b230561`.

The pinned foundation's `processor_config.json` declares eight modality configurations. Its `embodiment_id.json` maps those configurations to four shared state/action adapter banks:

| Adapter bank index | Processor configuration tags |
| --- | --- |
| 24 | `oxe_droid_relative_eef_relative_joint` |
| 25 | `real_g1_relative_eef_relative_joints` |
| 26 | `real_r1_pro_sharpa_relative_eef`, `real_r1_pro_sharpa_relative_eef_human`, `real_r1_pro_sharpa_relative_eef_maxinsights`, `real_r1_pro_sharpa_relative_eef_mecka` |
| 27 | `xdof_relative_eef_relative_joint`, `xdof_relative_eef_relative_joint_subtask` |

The full `embodiment_id.json` contains 52 tag names, including aliases for shared banks; this mapping is broader than the processor's eight configured modalities. Neither file declares an SO-100, SO-101 or SO_ARM head. The approved training plan uses `NEW_EMBODIMENT` with fresh SO-101 adapters at bank index 10.

The old 200k checkpoint is evaluation-only. Only its data/modality/processor settings and library-version evidence may inform this project; its weights never initialize training.

NVIDIA LICENSE and attribution files are retained. A modification notice is added in NOTICE. Source integrity hashes are recorded in `so101/provenance/upstream_gr00t_sha256.json` and were verified against the official pinned Git tree. Four ignored local SO100 artifacts (three camera captures and an evaluation lockfile) are excluded from that upstream manifest; their hashes and origin classification are recorded separately in `so101/provenance/local_so100_artifacts.json`.

The original upstream `.gitattributes` is retained in `so101/provenance/upstream_gitattributes.txt`. LFS filters are removed from this training repository so fresh clones require no Git LFS; unused upstream media pointers remain text and are outside the training path.

The three original deployment Dockerfiles are preserved byte-for-byte as text under `so101/provenance/upstream_dockerfiles/`, with their source paths and checksums in its manifest. The repository has one executable Dockerfile at its root for the SO-101 training workflow. Other upstream deployment documentation is retained for reference.

The official aarch64 TorchCodec 0.8.0 wheel is hydrated from its pinned upstream LFS payload and shipped as an ordinary Git binary. Its SHA-256 and download provenance are recorded in `so101/provenance/arm_torchcodec_wheel.json`. This wheel is needed to validate the upstream universal `uv.lock` (which resolves x86_64 and aarch64) without Git LFS; its existing version and lockfile hash are unchanged.
