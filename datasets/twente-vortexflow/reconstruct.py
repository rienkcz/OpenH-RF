# SPDX-License-Identifier: Apache-2.0
"""Example reconstruction script for the twente-vortexflow dataset of OpenH-RF.

Dataset link: https://huggingface.co/datasets/nvidia/OpenH-RF/tree/main/twente-vortexflow

B-mode reconstruction of single plane-wave flow-phantom channel data.

Each track has its own pipeline (``pipeline_<track>.yaml``); the script runs
them all on the chosen file and saves the results side by side in one PNG.

Requires zea>=0.1.6 (https://github.com/tue-bmd/zea), the library that does the
ultrasound processing here, together with one of its Keras backends (JAX,
PyTorch or TensorFlow). Installation instructions are at
https://zea.readthedocs.io/en/latest/installation.html.

Usage:
    python reconstruct.py
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import hdf5plugin
import numpy as np
import zea
from zea.ops import Beamform, Cast, Demodulate, EnvelopeDetect, LogCompress, Normalize

HERE = Path(__file__).parent

# Mapping from track label to the pipeline YAML shipped with this submission.
PIPELINE_YAMLS = {
    "short imaging pulse": HERE / "pipeline_short_imaging_pulse.yaml",
    "chirp": HERE / "pipeline_chirp.yaml",
}

XLIMS = (-0.08, 0.08)
ZLIMS = (0.05, 0.10)
GRID_SIZE_Z = 480

# --- Inputs -----------------------------------------------------------------
# Defaults stream straight from the published corpus. Swap any of these for a
# local path to run against your own copy.
ZEA_FILE = "hf://nvidia/OpenH-RF/twente-vortexflow/data/AcqData_PVoltage80_TVoltage3.4.hdf5"
FRAME = 10
OUT = HERE / "assets" / "reference_bmode.png"
OUT_2X1 = HERE / "assets" / "reference_mapping.png"


def build_pipeline() -> zea.Pipeline:
    """Build the default DAS B-mode pipeline.

    Chain: Cast → Demodulate → Beamform (DAS) → EnvelopeDetect → Normalize → LogCompress

    - Cast: converts raw_data to float32 (required before any float ops).
    - Demodulate: required for RF data (n_ch=1); no-op for IQ (n_ch=2).
    - Beamform: delay-and-sum with 100 patches for memory efficiency.
    - EnvelopeDetect / Normalize / LogCompress: standard B-mode display chain.
    """
    return zea.Pipeline(
        operations=[
            Cast(dtype="float32"),
            Demodulate(),
            Beamform(beamformer="delay_and_sum"),
            EnvelopeDetect(),
            Normalize(),
            LogCompress(),
        ],
        validate=False,
    )


def main() -> None:
    input_path = str(ZEA_FILE)
    if "://" not in input_path:
        candidate = Path(input_path)
        if not candidate.is_absolute():
            candidate = HERE / candidate
        if not candidate.exists():
            raise FileNotFoundError(
                f"Input file not found: {candidate}. Set ZEA_FILE to a file inside {HERE}."
            )
        input_path = str(candidate)

    zea.init_device()

    with zea.File(str(input_path)) as f:
        tracks = list(f.tracks)
        n_tracks = len(tracks)
        zea.visualize.set_mpl_style()
        fig, axes = plt.subplots(n_tracks, 1, figsize=(6 * n_tracks, 6))
        if n_tracks == 1:
            axes = [axes]
        track_panels = []

        for ax, track in zip(axes, tracks):
            label = track.label

            # Load or build the pipeline for this track.
            yaml_path = PIPELINE_YAMLS.get(label)
            if yaml_path and yaml_path.exists():
                # Load-and-run: use the saved YAML shipped with the submission.
                pipeline = zea.Pipeline.from_path(str(yaml_path))
            else:
                # Fallback: build the default pipeline in code.
                pipeline = build_pipeline()

            # Reconstruct one frame.
            params = track.load_parameters()
            # Apply fixed image limits before parameter prep so beamforming uses this FOV.
            params.xlims = XLIMS
            params.zlims = ZLIMS
            # Enforce square reconstruction sampling (same physical pixel size in x and z).
            params.grid_size_z = GRID_SIZE_Z
            x_span = params.xlims[1] - params.xlims[0]
            z_span = params.zlims[1] - params.zlims[0]
            params.grid_size_x = max(1, int(round(params.grid_size_z * x_span / z_span)))
            raw = track.data.raw_data[FRAME : FRAME + 1, ...]
            inputs = pipeline.prepare_parameters(params)
            outputs = pipeline(data=raw, **inputs)

            import keras

            recon = keras.ops.convert_to_numpy(outputs["data"])[0]
            extent_mm = [v * 1e3 for v in params.extent_imshow]

            image_frame = None
            if "image" in track.data:
                image_values = track.data.image.values[FRAME : FRAME + 1, ...]
                image_np = keras.ops.convert_to_numpy(image_values)
                image_frame = np.squeeze(image_np[0])
                # Rotate to match ultrasound orientation.
                image_frame = np.flipud(np.rot90(image_frame, 3))
            track_panels.append((label, recon, extent_mm, image_frame))

            ax.imshow(recon, cmap="gray", vmin=-50, vmax=0, extent=extent_mm, aspect="equal")
            ax.set_title(f"Track: {label}\nFile: {Path(input_path).name}, Frame {FRAME}")
            ax.set_xlabel("Lateral [mm]")
            ax.set_ylabel("Depth [mm]")

    plt.tight_layout()
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"Saved {OUT}")

    # Build requested 2x1 view:
    # top = short imaging pulse ultrasound
    # bottom = matching camera image.
    if track_panels:
        selected_panel = None
        for label, recon, extent_mm, image_frame in track_panels:
            if label.strip().lower() == "short imaging pulse":
                selected_panel = (label, recon, extent_mm, image_frame)
                break
        if selected_panel is None:
            selected_panel = track_panels[0]

        label, recon, extent_mm, image_frame = selected_panel
        fig2, axes2 = plt.subplots(2, 1, figsize=(7, 10))

        axes2[0].imshow(recon, cmap="gray", vmin=-50, vmax=0, extent=extent_mm, aspect="equal")
        axes2[0].set_title(f"Ultrasound ({label}) - frame {FRAME}")
        axes2[0].set_xlabel("Lateral [mm]")
        axes2[0].set_ylabel("Depth [mm]")

        if image_frame is not None:
            axes2[1].imshow(image_frame, cmap="gray", aspect="equal")
            axes2[1].set_title(f"Camera image - frame {FRAME}")
            axes2[1].set_xlabel("X [px]")
            axes2[1].set_ylabel("Y [px]")
        else:
            axes2[1].text(0.5, 0.5, "No camera image available", ha="center", va="center")
            axes2[1].set_title("Camera image unavailable")
            axes2[1].set_axis_off()

        fig2.tight_layout()
        Path(OUT_2X1).parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(OUT_2X1, dpi=150)
        plt.close(fig2)
        print(f"Saved {OUT_2X1}")


if __name__ == "__main__":
    main()
