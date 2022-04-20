

config = {
    'prefix_length' : 40,
    'clip_length' : 40,
    'prefix_size' : 512,
    'num_layers' : 8,
    'lr' : 2e-5,
    'epoch' : 10,
    'batch_size' : 32,
    'normalize_prefix' : True,
    'data' : '~/image-caption/oscar_split_ViT-B_32_train.pkl',
    'data_test' : '/data/usr/renyi/image-caption/oscar_split_ViT-B_32_test.pkl',
    'use_beam_search' : True,
    'weights_path' : './experiments/clip+gpt2_end/best/pytorch_model.bin',
    'out_path' : '/data/usr/renyi/image-caption/caption_result_fake.json',
}