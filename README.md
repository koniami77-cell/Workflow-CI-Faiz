# Workflow-CI (Faiz) — Telco Customer Churn

Workflow CI untuk re-training model **Telco Customer Churn** secara otomatis menggunakan MLflow Project + GitHub Actions, dan mem-build/push Docker image ke Docker Hub.

## Struktur

```
Workflow-CI/
├── .github/workflows/ci.yml
├── MLProject/
│   ├── MLProject                          # MLflow Project spec
│   ├── conda.yaml                         # environment
│   ├── modelling.py                       # skrip training untuk CI
│   └── telco_churn_preprocessing/         # data siap latih (train.csv, test.csv)
└── README.md
```

## Sebelum push ke GitHub

1. Buat repository baru **Public** bernama `Workflow-CI-Faiz` di akun GitHub [koniami77-cell](https://github.com/koniami77-cell).
2. Tambahkan GitHub Actions secrets pada repo (Settings → Secrets and variables → Actions):
   - `DOCKERHUB_USERNAME` → username Docker Hub Anda (huruf kecil semua, mis. `anfrb`).
   - `DOCKERHUB_TOKEN` → Access Token dari Docker Hub (Account Settings → Security → New Access Token), **bukan password**.
3. Push seluruh isi folder ini ke repo tersebut.

## Menjalankan secara lokal (opsional, untuk uji coba sebelum push)

```bash
cd MLProject
pip install mlflow==2.19.0 pandas numpy scikit-learn
mlflow run . --env-manager=local
```

## Catatan Docker Hub

Username Docker Hub **harus huruf kecil**. Jika akun GitHub Anda `koniami77-cell`, gunakan username Docker Hub yang sesuai (mis. `anfrb`) — ini yang harus diisi pada secret `DOCKERHUB_USERNAME`.

