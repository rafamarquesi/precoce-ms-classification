import os
import shutil
import argparse
from pathlib import Path
import warnings

import pandas as pd

import csv_treatments
import utils


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


def merge_cv_results():
    merge_cv_results_path = Path(os.getcwd()+'/runs/results/')

    if merge_cv_results_path.exists():
        merged_cv_results = pd.DataFrame()
        files_merged = 0
        for child in merge_cv_results_path.iterdir():
            if child.is_file() and child.name != '.gitkeep' and child.name.find('cv_results') != -1:
                print('\nMerging: {}'.format(child.name))
                merged_cv_results = pd.concat(
                    [merged_cv_results, csv_treatments.load_data(csv_path=child)])
                print(
                    '!!!---- Merged CV RESULTS SHAPE: {}'.format(merged_cv_results.shape))
                files_merged += 1
        if files_merged > 1:
            merged_cv_results = utils.move_cloumns_last_positions(data_frame=merged_cv_results, columns_names=[
                'params', 'split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score',
                'split5_test_score', 'split6_test_score', 'split7_test_score', 'split8_test_score', 'split9_test_score',
                'mean_test_score', 'std_test_score', 'rank_test_score'
            ])
            merged_cv_results = merged_cv_results.sort_values(
                'mean_test_score', ascending=False)
            # reports.informations(merged_cv_results)
            csv_treatments.generate_new_csv(
                data_frame=merged_cv_results,
                csv_path=str(merge_cv_results_path),
                csv_name='merged-cv_results-{}'.format(
                    utils.get_current_datetime())
            )
        else:
            warnings.warn(
                'Found only one file to merge CV RESULTS, not merged!')
    else:
        warnings.warn('Path to merge CV RESULTS does not exist!')


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

    # Merge CV results, before move the files
    merge_cv_results()

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
            warnings.warn(
                'Source path {} does not exist'.format(source_path))
