# Using Your Trained Model in the App

After training (e.g. with `scripts/train.py` or `make train`), use the saved checkpoint so the API runs in **real mode** instead of demo mode.

## 1. Get the checkpoint in the right place

The API looks for weights in `/app/weights` inside the container. Easiest: use a host `weights` folder and mount it.

- Create the directory (if you don’t have it):
  ```bash
  mkdir -p weights
  ```
- Put your checkpoint in it. The backend looks for (in order): `deepfake_detector.pt`, `best_model.pt`, `checkpoint.pt`, or any `*.pt` / `*.pth`.

  So either:
  - Copy your best checkpoint as `best_model.pt`:
    ```bash
    cp checkpoints/best_model.pt weights/
    ```
  - Or copy an epoch checkpoint (e.g. from training):
    ```bash
    cp checkpoints/checkpoint_epoch_26.pt weights/best_model.pt
    ```

If you use **Docker**: ensure the container sees this folder. With the provided override, `./weights` is mounted at `/app/weights`, so dropping the file in `weights/` is enough (see step 2).

## 2. Use the host `weights` folder in Docker (optional)

A `docker-compose.override.yml` can mount `./weights` into the backend and worker so you don’t need to copy files into a volume:

- Backend and celery-worker volumes include: `./weights:/app/weights`
- Restart after adding or changing files in `weights/`:
  ```bash
  docker compose down && docker compose up -d
  ```

If you don’t use the override, copy the file into the running backend container and restart:

```bash
docker cp weights/best_model.pt deepfake-backend:/app/weights/
docker restart deepfake-backend deepfake-celery-worker
```

## 3. Turn off demo mode

In `.env`:

```env
DEMO_MODE=false
```

Leave `MODEL_WEIGHTS_PATH` as `/app/weights` (default).

## 4. Start the platform

```bash
make up
```

Open the frontend (e.g. http://localhost:5173 or the URL shown in the Makefile). The API will load your checkpoint and run real inference.

## 5. Optional: export for production

To build production artifacts (state dict, ONNX, TorchScript) from a checkpoint:

```bash
# From project root; checkpoint path and output dir are configurable in the Makefile/script
make export-model
```

Use the checkpoint path that matches your training run (e.g. `checkpoints/best_model.pt`). Point the script’s `--output` at `weights/` if you want the exported file to be the one the API loads.

---

**If you don’t have a checkpoint file:**  
Your `checkpoints/` may only contain `training_results.json`. The actual `.pt` file is written during training (e.g. `checkpoints/best_model.pt` or `checkpoints/checkpoint_epoch_*.pt`). If it was on another machine or deleted, re-run training to get a new `best_model.pt`, then follow steps 1–4 above.
