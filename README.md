# Clinical ECG Noise Filter & Event Detection Pipeline

A production-grade Python pipeline designed to act as a **Triage Layer** for clinical Holter/telemetry data before it reaches an AI classification model (like a CNN).

## 🎯 The Problem
Medical AI models often generate false alarms (False Positives) because of messy data—specifically **Lead-Offs** (disconnected electrodes) and **Noise/Motion Artifacts**. If a Lead-Off is fed into an AI, it may misclassify a flatline as Ventricular Standstill.

## 🛠️ The Solution
This pipeline processes raw 24-hour `.edf` ECG recordings and algorithmically categorizes every second of data into:
1. **Clean:** Valid diagnostic QRS/Rhythm segments.
2. **Bradycardia:** Valid beats where HR < 60 BPM.
3. **Lead-Off (LD):** True flatlines or rail saturation (where the electrode detached).
4. **Noise:** High-frequency EMG or motion artifacts.

## 🚀 Key Features
* **Adaptive Filtering:** Uses a 0.5 Hz Butterworth high-pass filter to remove baseline wander without distorting QRS morphology.
* **Robust Lead-Off Logic:** Differentiates between high-amplitude R-peaks and true mathematical flatlines, preventing false alarm suppression.
* **Two-Tier Clinical PDF:** Automatically generates a multi-page Matplotlib PDF:
  * *Tier 1 (Page 1):* 24-hour macro timeline with color-coded event overlays.
  * *Tier 2 (Pages 2+):* 10-second micro zoom-ins proving the exact morphology (e.g., flatline for LD, prolonged R-R for Bradycardia).
* **Multi-File Validation:** Tested and validated across 7 MIT-BIH Long-Term Holter records (14–24 hours each).

## 💻 Tech Stack
* **Python 3.11+**
* **NeuroKit2** (R-peak detection & signal processing)
* **SciPy** (Butterworth filtering)
* **Matplotlib** (Clinical PDF generation)
* **MNE** (EDF file loading)

## ⚙️ How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place your `.edf` file in the directory.
3. Run the pipeline:
   ```bash
   python beatly_pipeline.py data.edf
   ```
4. Open `Beatly_ECG_Report.pdf` to view the clinical overview and event proof strips.
