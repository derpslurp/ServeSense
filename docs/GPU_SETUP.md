# Using a GPU for training and analysis

Training and ball detection will use your **NVIDIA GPU** automatically once PyTorch can see it. No script changes needed.

---

## 1. Check you have an NVIDIA GPU

- **Need:** NVIDIA GPU (GeForce, Quadro, RTX, etc.). AMD/Intel GPUs are not supported by this setup.
- In Windows: **Device Manager → Display adapters** — you should see an NVIDIA adapter.

---

## 2. Update your NVIDIA driver

- Download the latest driver for your GPU: [NVIDIA GeForce Drivers](https://www.nvidia.com/geforce/drivers/) or [NVIDIA Driver Download](https://www.nvidia.com/drivers).
- Newer drivers (e.g. 535+) usually work with current PyTorch CUDA builds.

---

## 3. Install PyTorch with CUDA

Right now you likely have the **CPU-only** PyTorch from `pip install torch`. Replace it with a **CUDA** build.

**Option A – CUDA 11.8** (works with most drivers):

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Option B – CUDA 12.1** (newer driver):

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Option C – CUDA 12.4** (latest, newest drivers):

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

If you’re unsure, try **cu118** first; if something fails, try **cu121**.

---

## 4. Verify the GPU is visible

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

You want:
- `CUDA available: True`
- `Device: <your GPU name>`

If you see `False` or an error, the driver may be too old or the CUDA version doesn’t match (try another `cuXXX` index above).

---

## 5. Run training again

No code changes are needed. Ultralytics will use the GPU when available:

```powershell
cd c:\Users\Warren\ServeSense
python scripts/evaluate_ball_model.py --train-epochs 50
```

You should see in the log something like:
- `torch-2.x.x+cu118` (or cu121/cu124) instead of `+cpu`
- `GPU_mem` in the epoch table showing usage instead of `0G`

Training should drop from **hours** on CPU to roughly **5–15 minutes** on a typical GPU.

---

## 6. Analysis (ServeSense app)

With PyTorch installed for CUDA, the **backend** (pose + ball detection) will also use the GPU when you run video analysis, so each analysis will be faster.

---

## Summary

| Step | Action |
|------|--------|
| 1 | Confirm NVIDIA GPU in Device Manager |
| 2 | Update NVIDIA driver from nvidia.com |
| 3 | `pip uninstall torch torchvision torchaudio -y` then `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` (or cu121/cu124) |
| 4 | `python -c "import torch; print(torch.cuda.is_available())"` → should print `True` |
| 5 | Run training again; it will use the GPU automatically |
