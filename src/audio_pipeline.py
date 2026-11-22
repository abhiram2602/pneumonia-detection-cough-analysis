"""Cough-audio feature extraction and auditable dataset inspection.
Research only; not a medical diagnostic device.
"""
import argparse, json
from pathlib import Path
import numpy as np
import librosa

def extract_features(path, sr=16000, duration=5):
    y, sr = librosa.load(path, sr=sr, duration=duration)
    if len(y) == 0:
        raise ValueError(f'Empty audio: {path}')
    y = librosa.util.fix_length(y, size=sr*duration)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    rms = librosa.feature.rms(y=y)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), delta.mean(axis=1), [rms.mean(), rms.std()]]).astype('float32')

def inspect(root, out):
    root=Path(root); out=Path(out);out.mkdir(parents=True,exist_ok=True)
    files=sorted(root.rglob('*.wav'))
    if not files:raise FileNotFoundError('No .wav recordings found')
    features=[];errors=[]
    for path in files:
        try:features.append((str(path),extract_features(path)))
        except Exception as exc:errors.append({'path':str(path),'error':str(exc)})
    if not features:raise ValueError('No readable audio files')
    X=np.stack([v for _,v in features]);np.savez_compressed(out/'audio_features.npz',X=X,paths=np.array([p for p,_ in features]))
    report={'files_found':len(files),'files_processed':len(features),'feature_dimensions':int(X.shape[1]),'errors':errors,'labels':'unavailable: all supplied WAVs are bronchitis; pneumonia and normal audio required for classification','classification_accuracy':None}
    (out/'execution_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data');p.add_argument('--output',default='outputs');a=p.parse_args();inspect(a.data,a.output)
