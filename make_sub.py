import click
from zipfile import ZipFile

@click.command()
@click.option('--weights-path', help='The path to the weights of the network.')
@click.option('--model-path', prompt='the path to the model')
def make_sub(weights_path, model_path):
  zipObj = ZipFile('sub.zip', 'w')
  zipObj.write(f'{weights_path}')
  zipObj.write(f'{model_path}')
  zipObj.close()
  
  
make_sub()
