# elastic moduli analysis
python3 fft_homogenization.py -r "GAN" -m "HM" -S 800 -s 800 -p 0 -o 'dry'
python3 fft_homogenization.py -r "GAN" -m "HM" -S 800 -s 400 -p 0 -o 'dry'
python3 fft_homogenization.py -r "GAN" -m "HM" -S 800 -s 200 -p 0 -o 'dry'
python3 fft_homogenization.py -r "GAN" -m "HM" -S 800 -s 100 -p 0 -o 'dry'
python3 fft_homogenization.py -r "GAN" -m "OT" -S 800 -s 800 -p 0 -o 'dry'
python3 fft_homogenization.py -r "GAN" -m "OT" -S 800 -s 400 -p 0 -o 'dry'
python3 fft_homogenization.py -r "GAN" -m "OT" -S 800 -s 200 -p 0 -o 'dry'
python3 fft_homogenization.py -r "GAN" -m "OT" -S 800 -s 100 -p 0 -o 'dry'

python3 fft_homogenization.py -r "GAN" -m "HM" -S 400 -s 400 -p 0 -o 'dry'
python3 fft_homogenization.py -r "GAN" -m "HM" -S 400 -s 200 -p 0 -o 'dry'
python3 fft_homogenization.py -r "GAN" -m "HM" -S 400 -s 100 -p 0 -o 'dry'
python3 fft_homogenization.py -r "GAN" -m "OT" -S 400 -s 400 -p 0 -o 'dry'
python3 fft_homogenization.py -r "GAN" -m "OT" -S 400 -s 200 -p 0 -o 'dry'
python3 fft_homogenization.py -r "GAN" -m "OT" -S 400 -s 100 -p 0 -o 'dry'

python3 fft_homogenization.py -r "GAN" -m "orig" -S 400 -s 400 -p 0 -o 'dry'

# Training CNN
# training CNN just based on R400 rocks
python3 -i train_cnn.py \
   --scan_dir rock_data/LR_dry_OT/ \
   --dat_file data/stiffness_OT_400_100_dry.dat \
   --epochs 1000 \
   --batch_size 8 \
   --scan_shape 100 100 100 \
   --lr 1e-3 \
   --train_ratio 0.8 \
   --stop_option 'test'

# training CNN just based on R800 rocks
python3 -i train_cnn.py \
   --scan_dir rock_data/HR_dry_OT/ \
   --dat_file data/stiffness_OT_800_100_dry.dat \
   --epochs 1000 \
   --batch_size 8 \
   --scan_shape 100 100 100 \
   --lr 1e-3 \
   --train_ratio 0.8 \
   --stop_option 'test'

# test on Benchmark 1 B1 data
python3 -i train_cnn.py \
   --scan_dir ../../../machine_learning/size_75/data/data_B1/skin_0/images/ \
   --dat_file ../../../machine_learning/size_75/data/data_B1/skin_0/stiffness.dat \
   --epochs 1000 \
   --batch_size 8 \
   --scan_shape 75 75 75 \
   --lr 1e-3 \
   --train_ratio 0.8 \
   --stop_option 'test'

# training composite CNN (R400 & R800 rocks)
python3 train_composite_cnn.py \
  --lr_scan_dir rock_data/LR_dry_OT/ \
  --lr_dat_file data/stiffness_OT_400_100_dry.dat \
  --hr_scan_dir rock_data/HR_dry_OT/ \
  --hr_dat_file data/stiffness_OT_800_100_dry.dat \
  --lr_epochs 1000 \
  --hr_epochs 100 \
  --batch_size 8 \
  --scan_shape 100 100 100 \
  --lr 1e-3 \
  --ft_lr 1e-4 \
  --train_ratio 0.8 \
  --stop_option test \
  --loss_avg_step 10 \
  --hr_subset 100 \

# CNN prediction
python3 -i eval_cnn.py \
   --scan_dir rock_data/LR_dry_OT/ \
   --dat_file data/stiffness_OT_400_100_dry.dat \
   --scan_shape 100 100 100 \
   --model_path model/rock_cnn_model.pt \
   --full_rock 400
# rock_cnn_model | composite_cnn_model

python3 -i eval_cnn.py \
   --scan_dir rock_data/HR_dry_OT/ \
   --dat_file data/stiffness_OT_800_100_dry.dat \
   --scan_shape 100 100 100 \
   --model_path model/rock_cnn_model.pt \
   --full_rock 800

python3 fft_homogenization.py -r "GAN" -m "CNN" -S 800 -s 100 -p 0 -o 'dry'
python3 fft_homogenization.py -r "GAN" -m "CNN" -S 400 -s 100 -p 0 -o 'dry'
# saturated rock analysis
# python3 fft_homogenization.py -r "GAN" -m "OT" -S 800 -s 800 -p 0 -o 'sat'
# python3 fft_homogenization.py -r "GAN" -m "OT" -S 800 -s 400 -p 0 -o 'sat'
# python3 fft_homogenization.py -r "GAN" -m "OT" -S 800 -s 200 -p 0 -o 'sat'
# python3 fft_homogenization.py -r "GAN" -m "OT" -S 800 -s 100 -p 0 -o 'sat'

# python3 fft_homogenization.py -r "GAN" -m "OT" -S 400 -s 400 -p 0 -o 'sat'
# python3 fft_homogenization.py -r "GAN" -m "OT" -S 400 -s 200 -p 0 -o 'sat'
# python3 fft_homogenization.py -r "GAN" -m "OT" -S 400 -s 100 -p 0 -o 'sat'

# plotting
python3 plot_mod_figures.py
# # gassmann equation analysis
# python3 validate_gassmann.py -r "GAN" -m "OT" -S 800 -s 800 
# python3 validate_gassmann.py -r "GAN" -m "OT" -S 800 -s 400
# python3 validate_gassmann.py -r "GAN" -m "OT" -S 800 -s 200
# python3 validate_gassmann.py -r "GAN" -m "OT" -S 800 -s 100

# python3 validate_gassmann.py -r "GAN" -m "OT" -S 400 -s 400
# python3 validate_gassmann.py -r "GAN" -m "OT" -S 400 -s 200
# python3 validate_gassmann.py -r "GAN" -m "OT" -S 400 -s 100