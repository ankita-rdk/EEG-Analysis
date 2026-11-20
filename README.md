# EEG Meditation State Analysis (`med_eeg.ipynb`)

**4-Channel Wearable-Montage EEG Meditation Classification & Benchmark Notebook**

This repository contains `med_eeg.ipynb`, a self-contained Jupyter/Colab notebook exploring **Mind-Wandering** vs. **Focused Meditation** classification on a 4-channel wearable-montage subset (`muse2_4ch`: `TP9`, `AF7`, `AF8`, `TP10`) derived from `256 Hz` (`128 Hz` Nyquist limit) EEG recordings (`ds001787` / processed shards)[cite: 1, 4].

---

## Repository Contents

* `med_eeg.ipynb` — End-to-end Colab/Jupyter notebook covering montage setup, signal conditioning, feature extraction (`GPUNeuroPhysExtractor`, filter-bank log-covariance / Riemannian features, and `NeuroWaveMed` experiments), benchmark evaluation, Google Drive checkpointing, and Gradio visualization[cite: 1, 4, 9, 10, 11, 12].
* `README.md` — Project overview, methodological notes, and summary of benchmark runs[cite: 1, 4].

---

## Methodological Notes

* **Simulated 4-Channel Montage:** `muse2_4ch` (`TP9`, `AF7`, `AF8`, `TP10`) is a 4-channel montage subset evaluated offline, not live recordings from physical Muse 2 hardware[cite: 1, 4].
* **Leakage & Split Sensitivity:** Overlapping sliding windows (`4 s` windows, `2 s` stride over `12 s` pre-probe lookbacks) and random/stratified splits inflate accuracy and AUC via shared window content and temporal autocorrelation[cite: 1, 4]. Honest evaluation requires **probe-grouped splits**, **per-probe prediction aggregation**, **chronological / Leave-One-Subject-Out (LOSO)** validation, **subject-level bootstrap CIs**, and **probe-block permutation p-values**[cite: 1].
* **Frequency & Artifact Boundaries:** At `256 Hz` (`128 Hz` Nyquist), `1–45 Hz` conditioning with a 4th-order bandpass filter attenuates above `45 Hz` (filter skirt) unless raw recordings are conditioned across a wider passband with full-cap artifact handling (`ICA` / amplitude checks) and without unlabeled fallback poisoning[cite: 1].

---

## Benchmark Snapshot (`muse2_4ch` Subset)

| Pipeline | Evaluation Protocol | Raw Acc (%) | Bal Acc (%) | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: |
| **Zero-Shot NeuroWaveMed v4** | 5-Fold GroupKFold (unseen subjects)[cite: 1] | 58.89 | 57.15 | 0.5599 |
| **AdamW Fine-Tuning (v4)** | 5-Fold Stratified (20% cal)[cite: 1] | 66.78 | 64.78 | 0.6975 |
| **Frozen v4 Embeddings + RBF-SVM** | 5-Fold Stratified (20% cal)[cite: 1] | 68.96 | 65.83 | 0.6821 |
| **1–80 Hz Filter-Bank + Shrinkage LDA** | Stratified (20% / 30% cal)[cite: 1, 4] | ~75.1–76.4 | ~68.4–70.3 | ~0.75–0.77 |
| **1–80 Hz Static LDA + LogReg Blend** | Strict Chronological (30% cal)[cite: 1, 4] | 61.27 | 57.32 | 0.6112 |

---

## How to Run

1. Open `med_eeg.ipynb` in **Google Colab** or **Jupyter**.
2. Mount Google Drive (if using Drive checkpoints under `NeuroWaveMed_Benchmarks`) and install required packages (`mne`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `torch`, `matplotlib`, `gradio`, `pyyaml`)[cite: 1, 3, 4, 8].
3. Run the cells sequentially from top to bottom. Keep `share=False` when launching Gradio locally or with sensitive/private EEG files[cite: 1, 7, 8].

---

## License

MIT License[cite: 2, 4]
