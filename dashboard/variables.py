"""
TODO Add module docstring
"""

import yaml

def read_variables(yaml_file):
    # read YAML file
    with open(yaml_file) as f:
        yaml_str = f.read()
    # load YAML dictionary
    yaml_dict = yaml.safe_load(yaml_str)
    # dictionary of input variables (parameters)
    input_variables = yaml_dict["input_variables"]
    # dictionary of output variables (objectives)
    output_variables = yaml_dict["output_variables"]
    return (input_variables, output_variables)
