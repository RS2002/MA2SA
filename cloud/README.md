# Cloud Job Assignment

## 1. Download data

Download these two inputs yourself from Alibaba's official sources:

- [Preprocessed 100K-job CSV](https://raw.githubusercontent.com/alibaba/clusterdata/3858992ca41bf92082e6d6522ddf0bebd14e838b/cluster-trace-gpu-v2020/simulator/traces/pai/pai_job_no_estimate_100K.csv).
- [Machine specifications](https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_machine_spec.tar.gz); extract `pai_machine_spec.csv` from the archive.
- [Dataset description and terms](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020).

Place the two CSV files under `data/raw/`.

## 2. Pre-train

```bash
python -m code.pretrain \
  --data-dir data/processed \
  --config configs/pretrain.json \
  --output runs/pretrain \
  --seed 0
```

## 3. Fine-tune

```bash
python -m code.finetune \
  --data-dir data/processed \
  --config configs/finetune.json \
  --checkpoint runs/pretrain/seed0/checkpoint_ep064.pt \
  --output runs/finetune \
  --seed 0
```
