# Trained Weights Directory

Place your trained G2P2C weights here after downloading from Kaggle.

## Naming Convention

```
g2p2c_adult_001_Actor.pth    # Actor weights for adult#001
g2p2c_adult_001_Critic.pth   # Critic weights for adult#001
g2p2c_adult_002_Actor.pth    # Actor weights for adult#002
g2p2c_adult_002_Critic.pth   # Critic weights for adult#002
...
g2p2c_adult_010_Actor.pth    # Actor weights for adult#010
g2p2c_adult_010_Critic.pth   # Critic weights for adult#010
```

## How to Get Weights

1. Run `kaggle_train_g2p2c.py` on Kaggle for each adult patient
2. Download the weights from `/kaggle/working/trained_weights/`
3. Place them in this folder
4. Run `streamlit run simulation_app.py` — only patients with weights will appear

## Patient ID Reference

| Patient | patient_id | Weight Files |
|---------|-----------|--------------|
| adult#001 | 20 | g2p2c_adult_001_*.pth |
| adult#002 | 21 | g2p2c_adult_002_*.pth |
| adult#003 | 22 | g2p2c_adult_003_*.pth |
| adult#004 | 23 | g2p2c_adult_004_*.pth |
| adult#005 | 24 | g2p2c_adult_005_*.pth |
| adult#006 | 25 | g2p2c_adult_006_*.pth |
| adult#007 | 26 | g2p2c_adult_007_*.pth |
| adult#008 | 27 | g2p2c_adult_008_*.pth |
| adult#009 | 28 | g2p2c_adult_009_*.pth |
| adult#010 | 29 | g2p2c_adult_010_*.pth |
