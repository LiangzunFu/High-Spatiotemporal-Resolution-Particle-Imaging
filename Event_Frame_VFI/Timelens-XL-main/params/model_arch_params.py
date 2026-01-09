from easydict import EasyDict as ED

model_arch_config = ED()

# model_arch_config.Expv8_large = ED()
# model_arch_config.Expv8_large.define_model = ED()
# model_arch_config.Expv8_large.define_model.type = 'FinalBidirectionAttenfusion'  # UNetPSDecoderRecurrent #UNetDecoderRecurrent
# model_arch_config.Expv8_large.define_model.base_channel = 32
# model_arch_config.Expv8_large.define_model.echannel = 128
# model_arch_config.Expv8_large.define_model.num_decoder = 8
# model_arch_config.Expv8_large.define_model.img_chn = 6  # 6 for two image, 26 for image and voxel
# model_arch_config.Expv8_large.define_model.ev_chn = 2
# model_arch_config.Expv8_large.define_model.num_encoders = 3
# model_arch_config.Expv8_large.define_model.base_num_channels = 32
# model_arch_config.Expv8_large.define_model.out_chn = 3

model_arch_config.Expv8_large = ED()
model_arch_config.Expv8_large.define_model = ED()
model_arch_config.Expv8_large.define_model.type = 'FinalBidirectionAttenfusion'  # UNetPSDecoderRecurrent #UNetDecoderRecurrent
# large model config
model_arch_config.Expv8_large.define_model.base_channel = 32
model_arch_config.Expv8_large.define_model.echannel = 128
model_arch_config.Expv8_large.define_model.num_decoder = 8
# base model config
# model_arch_config.Expv8_large.define_model.base_channel = 16
# model_arch_config.Expv8_large.define_model.echannel = 64
# model_arch_config.Expv8_large.define_model.num_decoder = 4
model_arch_config.Expv8_large.define_model.img_chn = 6  # 6 for two image, 26 for image and voxel
