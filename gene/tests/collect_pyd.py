"""


"""

from pathlib import Path
import os
import re
import shutil


excludes = ".+[.py]$|.+[.c]$|.+[.ui]$"


def get_pyd_module(module_name, project_name, excludes=None):
    file_list = []
    for root, dirs, files in os.walk(module_name):
        for file in files:
            if re.match(excludes, file) and not re.match("__init__", file) and not re.match("_imgs_rc.py", file):
                continue
            file_list.append(os.path.join(root, file))
    return file_list


def copy_pyd_package(target_dir, project_name, file_list):
    target_dir = abs_path.joinpath(target_dir).joinpath(project_name)
    os.makedirs(target_dir, exist_ok=True)
    for file in file_list:
        sub_file = str(file).split(project_name)[-1].strip("\\")
        target_file = target_dir.joinpath(sub_file)
        os.makedirs(target_file.parent.__str__(), exist_ok=True)
        print(file, target_file.__str__())
        shutil.copy(file, target_file)

def copy_packing_file(target_dir, abs_path: Path):
    filenames = ["setup.py", "main_gui.py"]
    for name in filenames:
        orgin_file = abs_path.joinpath(name)
        target_file = abs_path.joinpath(target_dir).joinpath(name)
        print(orgin_file, target_file.__str__())
        shutil.copy(orgin_file, target_file)

if __name__ == '__main__':
    project_name1 = "pyQCat"
    abs_path = Path(__file__).parent.parent
    project_path1 = abs_path.joinpath(project_name1)
    project_name2 = "pyqcat_visage"
    project_path2 = abs_path.joinpath(project_name2)
    # os.removedirs(".build_pyd")
    target_dir = ".build_pyd"
    copy_pyd_package(target_dir, project_name1, get_pyd_module(project_path1, project_name1, excludes))
    copy_pyd_package(target_dir, project_name2, get_pyd_module(project_path2, project_name2, excludes))
    # copy_packing_file(target_dir, abs_path)
