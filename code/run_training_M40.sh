
# Train the 4D network on M40 dataset
python wrapper_modelnet40_train.py --dataset vanillaM40Dataset \
--gpu 0 \
--log_dir m40_wahl_conv4d/ \
--batch_size 80 \
--max_epoch 2000 \
--conv_dim 4 \
--desc_type 1

