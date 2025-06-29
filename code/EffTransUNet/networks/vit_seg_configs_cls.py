import ml_collections


def get_b16_config():
    config = ml_collections.ConfigDict()
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim             = 3072
    config.transformer.num_heads           = 12
    config.transformer.num_layers          = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate        = 0.1
    config.classifier       = 'cls'
    config.n_classes        = 4
    config.representation_size = None
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels    = [136, 48, 32, 0]
    config.n_skip           = 0
    config.activation       = 'softmax'
    config.head_hidden_dim  = 512
    config.pretrained_path      = None
    config.resnet_pretrained_path = None
    config.patches = ml_collections.ConfigDict()
    config.patches.size = None
    config.patches.grid = None
    config.resnet = None

    return config


def get_r50_b16_config():
    config = get_b16_config()
    config.patches.grid = None
    config.classifier       = 'cls'
    return config


def get_testing():
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


