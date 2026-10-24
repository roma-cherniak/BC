start the test file  = venv\Scripts\python test_recordes.py


1)  **data_collection** - folder with scripts that run communication with Autodrive simulator and drive it with find_gap script 
2) All data that was collected by scripts are stored at data folder in .npz format 
3) data_prep work with data and make them to right training format
4) simple_model folder with basik MLP model that trained on my data , its simple  and just for start 



MLP world model (40 epochs, model.pt best val at epoch 38)
- one step (0.05 s): test/copy = 0.24
- rollout: brain/copy = 0.25 @0.05 s, 0.35 @0.5 s, 0.67 @1 s
- crossing ~1.1 s, after that a few imaginations explode (values < 0 or > 1)


JEPA world model on lidar (encoder 271->256->32, predictor 35->256->256->32, 161,088 params)
- without SIGReg: full collapse in 10 epochs (spread 1.7e-06, val loss ~4e-10)
- with SIGReg (weight 0.09, 50 epochs): spread 1.01, sigreg 1.59, one step val/copy = 0.24 (in latent space)
- probes R2 on test, trained / random encoder:
  - linear: front 0.16 / 0.90, right 0.30 / 0.75, left 0.16 / 0.96, steering 0.19 / 0.60, speed ~0 (control)
  - MLP probe: front 0.99 / 0.98, right 0.98 / 0.89, left 0.995 / 0.98, steering 0.83 / 0.69,
    track position x / y 0.99 / 0.99 (random 0.93 / 0.86)
  - -> information is kept but "folded" (nonlinear), the encoder learned where the car is on the track
  - (old 10-epoch encoder, file lost: linear probes front 0.90, right 0.85, left 0.95, steering 0.68)
- rollout in latent space (500 random safe test starts), MSE brain vs copy:
  - no guard: better than copy until ~0.75 s, then explodes (5.8 @1 s), imagined |z| grows 5.5 -> 7.5
    while real |z| stays ~sqrt(32) = 5.66
  - guard B (rescale z to length sqrt(32) after every step): better than copy for the whole 2 s,
    ~0.95 vs ~1.9 @1 s, ~1.55 vs 2.0 @2 s
  - "2" = error between two random descriptions (random level)




what data contain packet = ['V1 Throttle', 'V1 Steering', 'V1 Speed', 'V1 Encoder Ticks', 'V1 Encoder Angles', 'V1 Position', 'V1 Orientation Quaternion', 'V1 Orientation Euler Angles', 'V1 Angular Velocity', 'V1 Linear Acceleration', 'V1 LIDAR Scan Rate', 'V1 LIDAR Range Array', 'V1 Front Camera Image', 'V1 Lap Count', 'V1 Lap Time', 'V1 Last Lap Time', 'V1 Best Lap Time', 'V1 Collisions']



