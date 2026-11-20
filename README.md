# EEG Meditation State Analysis (`neurowave_meditation_eeg`)

**4-Channel Wearable-Montage EEG Meditation Classification & Benchmark Notebook**

This repository contains `neurowave_meditation_eeg.ipynb`, a self-contained Jupyter/Colab notebook exploring **Mind-Wandering** vs. **Focused Meditation** classification on a 4-channel wearable-montage subset (`muse2_4ch`: `TP9`, `AF7`, `AF8`, `TP10`) derived from `256 Hz` (`128 Hz` Nyquist limit) EEG recordings (`ds001787` / processed shards).

---

## Repository Contents

* `neurowave_meditation_eeg.ipynb` — End-to-end Colab/Jupyter notebook covering montage setup, signal conditioning, feature extraction (`GPUNeuroPhysExtractor`, filter-bank log-covariance / Riemannian features, and `NeuroWaveMed` experiments), benchmark evaluation, Google Drive checkpointing, and Gradio visualization.
* `README.md` — Project overview, methodological notes, and summary of benchmark runs.

---

## Methodological Notes

* **Simulated 4-Channel Montage:** `muse2_4ch` (`TP9`, `AF7`, `AF8`, `TP10`) is a 4-channel montage subset evaluated offline, not live recordings from physical Muse 2 hardware.
* **Window Overlap & Leakage:** Sliding `4 s` windows with a `2 s` stride across `12 s` pre-probe segments overlap by `50%` and share labels within the same probe event, so random or stratified splits place near-identical windows in both calibration and test sets and inflate accuracy/AUC (~`74–76%` vs. ~`61%` under chronological splits).
* **Probe Grouping & Per-Probe Aggregation:** Because multiple windows come from the same thought-probe event (effective sample size is on the order of probes rather than individual overlapping windows), splits must be grouped by `probe_id` (chronologically or via Leave-One-Subject-Out) with predictions aggregated per probe before computing metrics and balancing class skew across subjects.
* **Bootstrap Confidence Intervals & Probe-Block Permutation Testing:** Subject-level **bootstrap 95% confidence intervals** quantify uncertainty across subjects rather than relying on a single fold or point estimate, while **probe-block permutation tests** (shuffling labels by probe block within subject) test whether observed performance exceeds chance under realistic probe structure.
* **Frequency & Artifact Boundaries:** At `256 Hz` (`128 Hz` Nyquist), `1–45 Hz` conditioning with a 4th-order bandpass filter attenuates above `45 Hz` (filter skirt) unless raw recordings are conditioned across a wider passband with full-cap artifact handling (`ICA` / amplitude checks) and without unlabeled fallback poisoning.

---

## Benchmark Snapshot (`muse2_4ch` Subset)

| Pipeline | Evaluation Protocol | Raw Acc (%) | Bal Acc (%) | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: |
| **Zero-Shot NeuroWaveMed v4** | 5-Fold GroupKFold (unseen subjects) | 58.89 | 57.15 | 0.5599 |
| **AdamW Fine-Tuning (v4)** | 5-Fold Stratified (20% cal) | 66.78 | 64.78 | 0.6975 |
| **Frozen v4 Embeddings + RBF-SVM** | 5-Fold Stratified (20% cal) | 68.96 | 65.83 | 0.6821 |
| **1–80 Hz Filter-Bank + Shrinkage LDA** | Stratified (20% / 30% cal) | ~75.1–76.4 | ~68.4–70.3 | ~0.75–0.77 |
| **1–80 Hz Static LDA + LogReg Blend** | Strict Chronological (30% cal) | 61.27 | 57.32 | 0.6112 |

---

## How to Run

1. Open `neurowave_meditation_eeg.ipynb` in **Google Colab** or **Jupyter**.
2. Mount Google Drive (if using Drive checkpoints under `NeuroWaveMed_Benchmarks`) and install required packages (`mne`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `torch`, `matplotlib`, `gradio`, `pyyaml`).
3. Run the cells sequentially from top to bottom. Keep `share=False` when launching Gradio locally or with sensitive/private EEG files.

---

## License

MIT License
