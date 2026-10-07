# Upstream provenance

Source: https://github.com/NVIDIA/Isaac-GR00T.git

Pinned commit: `51d4c89f72fda44cbf77285c6a8114b52676b8a1`. The source working tree had no local differences before restructuring. NVIDIA's `gr00t/` sources are retained unchanged; project adaptations belong in `so101/` and runtime patches.

Official base: `nvidia/GR00T-N1.7-3B`, named by the pinned README, revision `2fc962b973bccdd5d8ce4f67cc63b264d6886495`.

Backbone: `nvidia/Cosmos-Reason2-2B`, revision `9ce19a195e423419c349abfc86fd07178b230561`.

The base's pretrained tags are `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`, `XDOF`, `XDOF_SUBTASK`, `REAL_G1`, `REAL_R1_PRO_SHARPA`, `REAL_R1_PRO_SHARPA_HUMAN`, `REAL_R1_PRO_SHARPA_MAXINSIGHTS`, and `REAL_R1_PRO_SHARPA_MECKA`. None is SO-100/SO-101-compatible. This project uses `NEW_EMBODIMENT` with fresh SO-101 adapters.

The old 200k checkpoint is evaluation-only. Only its data/modality/processor settings and library-version evidence may inform this project; its weights never initialize training.

NVIDIA LICENSE and attribution files are retained. A modification notice is added in NOTICE. Source integrity hashes are recorded in `so101/provenance/upstream_gr00t_sha256.json`.

The original upstream `.gitattributes` is retained in `so101/provenance/upstream_gitattributes.txt`. LFS filters are removed from this training repository so fresh clones require no Git LFS; unused upstream media pointers remain text and are outside the training path.

The three original deployment Dockerfiles are preserved byte-for-byte as text under `so101/provenance/upstream_dockerfiles/`, with their source paths and checksums in its manifest. The repository has one executable Dockerfile at its root for the SO-101 training workflow. Other upstream deployment documentation is retained for reference.
