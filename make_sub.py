import click

@click.command()
@click.option('--weights-path', help='The path to the weights of the network.')
@click.option('--model-path', prompt='the path to the model')
def make_sub(weights_path, model_path):
  ...
  
make_sub()
