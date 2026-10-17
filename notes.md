start the test file  = venv\Scripts\python test_recordes.py


1)  **data_collection** - folder with scripts that run communication with Autodrive simulator and drive it with find_gap script 
2) All data that was collected by scripts are stored at data folder in .npz format 
3) data_prep work with data and make them to right training format
4) simple_model folder with basik MLP model that trained on my data , its simple  and just for start 



MLP world model (40 epochs, model.pt best val at epoch 38)
- one step (0.05 s): test/copy = 0.24
- rollout: brain/copy = 0.25 @0.05 s, 0.35 @0.5 s, 0.67 @1 s
- crossing ~1.1 s, after that a few imaginations explode (values < 0 or > 1)




what data contain packet = ['V1 Throttle', 'V1 Steering', 'V1 Speed', 'V1 Encoder Ticks', 'V1 Encoder Angles', 'V1 Position', 'V1 Orientation Quaternion', 'V1 Orientation Euler Angles', 'V1 Angular Velocity', 'V1 Linear Acceleration', 'V1 LIDAR Scan Rate', 'V1 LIDAR Range Array', 'V1 Front Camera Image', 'V1 Lap Count', 'V1 Lap Time', 'V1 Last Lap Time', 'V1 Best Lap Time', 'V1 Collisions']



