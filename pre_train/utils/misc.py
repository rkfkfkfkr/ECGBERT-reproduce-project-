import os
import yaml
import mergedeep

def load_yaml(filename: str):
    with open(filename) as f:
        d = yaml.load(f, Loader=yaml.FullLoader)
    return d

def yaml_load(yaml_path, override_yaml_path):
    assert os.path.isfile(yaml_path), f"No yaml configuration file found at {yaml_path}"
    params = load_yaml(yaml_path)

    if override_yaml_path:
        assert os.path.isfile(override_yaml_path), f"No yaml configuration file found at {override_yaml_path}"
        params = mergedeep.merge(params, load_yaml(override_yaml_path))

    return params