import os
import yaml
import click

from Utils.preprocessing_utils import hdf5_to_pyg_events

@click.command()
@click.argument('config', type=str, required=True)
def main(config):
    with open(config, 'r') as stream:
        config = yaml.safe_load(stream)
    for subdir in config["subdirs"]:
        input_subfile = os.path.join(config["input_dir"], f"{subdir}.h5")
        output_subdir = os.path.join(config["output_dir"], subdir)
        hdf5_to_pyg_events(input_subfile, output_subdir, config["feature_scales"])



if __name__ == "__main__":
    main()