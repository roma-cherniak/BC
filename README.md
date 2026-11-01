# BC: lidar JEPA for the AutoDRIVE car

Run every command from the `BC` folder. One sim at a time, and never the sim and a training together (4 GB GPU).

## One command for every job

| Job | Command |
|---|---|
| Collect trips in the sim (start the practice sim first) | `venv\Scripts\python -m data_collection.collection_random` |
| Check the collected trips | `venv\Scripts\python -m checks.check_data` |
| Split trips into train / val / test | `venv\Scripts\python -m data_prep.split_data` |
| Make the pairs `data/*_pairs.npz` | `venv\Scripts\python -m data_prep.prepare_data` |
| Train a new JEPA (change `RUN` in the file first) | `venv\Scripts\python -m training.train_jepa` |
| Train its probe | `venv\Scripts\python -m training.train_zond <run>` |
| Report of one model | `venv\Scripts\python -m checks.run_all <run>` |
| All models side by side | `venv\Scripts\python -m checks.compare` |
| Drive in the sim (set `MODEL` and `TRACK` in the file first) | `venv\Scripts\python -m driver_lidar.drive_jepa` |

`<run>` is a folder name in `runs/`, for example `k10_50ep`.

## Folders

| Folder | What is inside |
|---|---|
| `config.py` | the numbers several files must agree on |
| `core/` | shared code: sensors, data, models, imagination, planner, sigreg |
| `data_collection/` | talking to the sim and recording trips |
| `data_prep/` | split and pairs |
| `training/` | train_jepa, train_zond |
| `checks/` | the checks and the report (`run_all`, `compare`) |
| `driver_lidar/` | the driver for the sim |
| `runs/<run>/` | one trained model: encoder, predictor, zond, loss, settings, reports, drive logs |
| `experiments/` | one-off experiments; fix their imports only if you run them again |
| `archive/` | stage A (MLP) and old files, not used any more |
