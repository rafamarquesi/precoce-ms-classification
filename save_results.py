import os
import shutil
import argparse
from pathlib import Path


def get_active_branch_name():
    head_dir = Path(".") / ".git" / "HEAD"
    with head_dir.open("r") as f:
        content = f.read().splitlines()

    for line in content:
        if line[0:4] == "ref:":
            return line.partition("refs/heads/")[2]


def move_files(source_path, destination_path):
    # Move the content of source to destination
    for child in source_path.iterdir():
        if child.is_file() and child.name != '.gitkeep':
            print('Moving file {} to {}'.format(
                child.name, str(destination_path)))
            dest = shutil.move(str(child), str(destination_path))
        elif child.is_dir():
            destination = Path(str(destination_path.absolute())+'/'+child.name)
            if not destination.exists():
                print('Creating directory {}'.format(destination))
                destination.mkdir(parents=True, exist_ok=True)
            else:
                print('Directory already exists {}'.format(destination))
            move_files(child, destination)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-file', help='File to be moved', type=str)
    args = parser.parse_args()

    file_path = Path(os.getcwd()+'/'+args.file)

    folders_with_results = ['logs', 'plots', 'runs']

    print('Branch name: {}'.format(get_active_branch_name()))

    results_dir = 'results/'+get_active_branch_name()

    # Create a directory to save the results
    out = Path(os.getcwd()+'/'+results_dir)
    if not out.exists():
        print('Creating directory {}'.format(out))
        out.mkdir(parents=True, exist_ok=True)
    else:
        print('Directory already exists {}'.format(out))

    # Move the file to the results directory
    if file_path.is_file():
        print('Moving file {} to {}'.format(
            file_path.name, str(out)))
        dest = shutil.move(str(file_path), str(out))

    for path in folders_with_results:
        source_path = Path(os.getcwd()+'/'+path)
        if source_path.exists():
            destination_path = Path(os.getcwd()+'/'+results_dir+'/'+path)
            if not destination_path.exists():
                print('Creating directory {}'.format(destination_path))
                destination_path.mkdir(parents=True, exist_ok=True)
            else:
                print('Directory already exists {}'.format(destination_path))
            move_files(source_path, destination_path)
        else:
            raise Exception(
                'Source path {} does not exist'.format(source_path))
