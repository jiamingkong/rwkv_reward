import types

RWKV_CONFIGS = {
    "430M": {
        "n_layer": 24,
        "n_embd": 1024,
    }
}

def get_config(model_name):
    """
    Locates the configuration for a given model name.
    """
    for k, v in RWKV_CONFIGS.items():
        if k in model_name:
            args = types.SimpleNamespace(**v)
            args.MODEL_NAME = model_name
            return args
