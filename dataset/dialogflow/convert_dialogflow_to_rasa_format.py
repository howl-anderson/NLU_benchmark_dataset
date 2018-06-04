import functools
import os
import pathlib
import string
import tempfile
import zipfile

from joblib import Parallel, delayed

from rasa_nlu.convert import convert_training_data

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

input_path = os.path.join(CURRENT_DIR, "dataset/dialogflow/weather/Weather")
output_path = os.path.join(CURRENT_DIR, "dataset/dialogflow/weather/RASA_format_dataset/data.json")

CURRENT_PATH = pathlib.Path(CURRENT_DIR)


def convert_alien_format_dataset(language, input_path, output_path):
    convert_training_data(
        data_file=input_path,
        out_file=output_path,
        output_format="json",
        language=language.lower()
    )


def convert_dataset(language, input_path, output_path, tmp_dir=pathlib.Path('./_tmp/raw_output')):
    tmp_dir.absolute().mkdir(exist_ok=True, parents=True)

    working_dir = tempfile.mkdtemp(dir=tmp_dir.absolute())

    data_file = os.path.join(working_dir, 'data.json')

    convert_alien_format_dataset(language, input_path, data_file)

    zip_ref = zipfile.ZipFile(output_path, 'w')
    zip_ref.write(data_file, arcname='data.json')
    zip_ref.close()


def get_input_dir(input_path_obj, tmp_dir=pathlib.Path('./_tmp/input_extracted')):
    input_file = str(input_path_obj.resolve())
    tmp_dir.mkdir(exist_ok=True, parents=True)

    working_dir = tempfile.mkdtemp(dir=tmp_dir.absolute())

    zip_ref = zipfile.ZipFile(input_file, 'r')
    zip_ref.extractall(working_dir)
    zip_ref.close()

    return working_dir


def get_output_file(input_path_obj):
    input_file = input_path_obj.resolve()
    output_file_parts = list(input_file.parts)
    output_file_parts[-2] = 'rasa_format'
    output_file = os.path.join(*output_file_parts)

    return str(output_file)


def get_language(input_path_obj):
    input_file = input_path_obj.resolve()
    file_name = input_file.parts[-1]
    language, _ = os.path.splitext(file_name)

    return language


# convert_alien_format_dataset(input_path, output_path)

raw_datasets = CURRENT_PATH.glob('**/dialogflow_format/*.zip')
input_output_of_converter = map(
    lambda x: (get_language(x), get_input_dir(x), get_output_file(x)),
    raw_datasets
)


for i in input_output_of_converter:
    convert_dataset(*i)
