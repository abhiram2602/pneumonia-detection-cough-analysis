# Enhanced Pneumonia Detection Through Cough Analysis Techniques

Academic capstone reproduction scaffold. The original report and poster describe cough-audio preprocessing, MFCC features and 1D CNN binary classification (pneumonia vs normal). The poster claims **92% accuracy**, and a classification report screenshot shows **~91% on 243 samples**. These results have **not** been independently reproduced.

## Verified execution

The uploaded `bronwav.rar` contains 50 bronchitis WAV recordings, **not** a labeled pneumonia-vs-normal dataset. The included feature pipeline has been run against these real files, producing `outputs/audio_features.npz` and `outputs/execution_report.json`. A medically meaningful pneumonia classifier cannot be trained or validated on this single-class bronchitis archive.

## Run

```bash
pip install -r requirements.txt
python src/audio_pipeline.py --data data --output outputs
```

Extract your legally shareable audio into `data/`. For pneumonia classification, provide separate **pneumonia and normal** recordings with patient identifiers and clinically verified labels; use patient-grouped train/test splits to prevent leakage. Do not interpret model output as medical advice.

## Reproduction notes

The original notebooks contain hard-coded Colab paths and one contains a Kaggle API token. The notebooks are intentionally **not** republished until that token is revoked and all secrets are removed. The project also includes an alternative respiratory sound database notebook, which is a different dataset and cannot establish the poster's exact accuracy.

## Source materials

Team capstone: V. Sathvika, Abdul Mannan Junaid, S. Abhi Ram, V. Akhila Reddy, K. Shivashanker Reddy. Preserve team attribution and obtain permission before public redistribution of recordings or reports.
