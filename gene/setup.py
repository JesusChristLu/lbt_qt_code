"""
-------build exe for windows with nuitka--------------
python setup.py bdist_nuitka

-------build exe for windows with cz_freeze--------------
python setup.py build_exe
Deprecated because of child process issues!

-------build cython .so or pyd-----------------
python setup.py build_ext

-------build package .whl----------------
python setup.py bdist_wheel
python setup.py sdist bdist_wheel

"""
import os
import re
import sys
from datetime import datetime
from setuptools import setup, find_packages

# check python version
python_major_version, *_ = sys.version_info

if python_major_version != 3:
    raise EnvironmentError('PyQCat-visage only support python 3')

DESCRIPTION = "pyQCat is a python framework for working with OriginQ quantum computers at the level of experiment " \
              "modules. pyqcat-visage provides friendly user interface for users to do physical quantum experiments."


# get long description from README.rst
# try:
#     with open("README.md") as fin:
#         LONG_DESCRIPTION = fin.read()
# except IOError('read README.md error'):
#     LONG_DESCRIPTION = None


# get build args
def is_build_pyd():
    if len(sys.argv) > 1:
        argv = sys.argv[1]
        if argv == "build_ext":
            return True
    return False


# get version
def get_version(version_tuple):
    """Return the version tuple as a string, e.g. for (0, 10, 7),
    return '0.10.7'.
    """
    return ".".join(map(str, version_tuple))


init = os.path.join(os.path.dirname(__file__), "pyqcat_visage", "__init__.py")
version_line = list(filter(lambda l: l.startswith("VERSION"), open(init)))[0]
VERSION = get_version(eval(version_line.split("=")[-1]))

# set classifiers
CLASSIFIERS = [
    'Development Status :: 2 - Beta',
    'Intended Audience :: Developers',
    'Topic :: DataCenter',
    "Operating System :: Microsoft :: Windows",
    # 'License :: OSI Approved :: MIT License',
    "Programming Language :: Python :: 3 :: Only",
    'Programming Language :: Python :: 3.9',
    "Programming Language :: Python :: Implementation :: CPython"
]

excludes = ['build', 'dist', 'docs', 'venv', 'venv3', "test", ".log"]

package_datas = {
    # "pyqcat-courier": []
}


# build pyd files list
def to_pyd_module(module_name, excludes=None):
    file_list = []
    for root, dirs, files in os.walk(module_name):
        for file in files:
            if re.findall(excludes, file):
                continue
            file_list.append(os.path.join(root, file))
    return file_list


def get_pyd_list():
    pyd_list = [
        # "./pyQCat/instrument/instrument_aio.py",
    ]

    excludes = "init|\.pyd|\.c"

    # pyd_list.extend(to_pyd_module("./pyQCat/experiment", excludes))
    # pyd_list.extend(to_pyd_module("./pyQCat/pulse", excludes))
    # pyd_list.extend(to_pyd_module("./pyQCat/database", excludes))
    excludes = "__init__|\.pyc|\.c|\.so|\.pxd|\.pyx|\.ini|\.dll|\.json|\.lib|\.yaml|\.exe|\.a|\.h|\.qrc|\.png|\.svg|\.css|\.qss|qss_template|\.ui|_imgs_rc|fonts_rc|\.md|\.inc|\.txt|\.npz|\.ttf|\.dat"
    pyd_list.extend(to_pyd_module("./pyQCat", excludes))
    pyd_list.extend(to_pyd_module("./pyqcat_visage", excludes))
    return pyd_list


def get_requirements():
    require_list = []
    with open("./requirements.txt", "r", encoding="utf-8") as fp:
        data = fp.readlines()
    for line in data:
        if not line.startswith("#"):
            require = line.strip("\n").strip(" ")
            if require:
                require_list.append(require)
    return require_list


if is_build_pyd():
    from distutils.core import setup
    from Cython.Build import cythonize
    pyd_list = get_pyd_list()
    setup(
        name="pyqcat-visage extensions",
        ext_modules=cythonize(get_pyd_list(), language_level=3)
    )

elif len(sys.argv) > 2 and sys.argv[2] == "bdist_wheel" or \
        len(sys.argv) == 2 and sys.argv[1] == "bdist_wheel":

    def delete_py_list_file(to_pyd_list):
        for file in to_pyd_list:
            try:
                os.remove(file)
            except:
                pass


    delete_py_list_file(get_pyd_list())

    # setup config
    setup(
        name="pyqcat-visage",
        version=VERSION,
        author="Origin Quantum Development Team",
        author_email="shq@originqc.com",
        # url="http://PyQCat.org/",
        # download_url="https://github.com/M1racleShih/PyQCat/tree/master",
        # license="MIT",
        packages=find_packages(exclude=excludes),
        python_requires="==3.9.*",
        include_package_data=True,
        package_data=package_datas,
        description=DESCRIPTION,
        # long_description=LONG_DESCRIPTION,
        platforms=["win_amd64"],
        classifiers=CLASSIFIERS,
        install_requires=get_requirements()
    )
elif len(sys.argv) > 2 and sys.argv[2] == "build_exe" or \
        len(sys.argv) == 2 and sys.argv[1] == "build_exe":
    import shutil
    from pathlib import Path
    from cx_Freeze import Executable, setup

    project_path = Path(__file__).absolute().parent
    try:
        from cx_Freeze.hooks import get_qt_plugins_paths
    except ImportError:
        get_qt_plugins_paths = None

    include_files = []
    if get_qt_plugins_paths:
        # Inclusion of extra plugins (since cx_Freeze 6.8b2)
        # cx_Freeze imports automatically the following plugins depending of the
        # use of some modules:
        # imageformats, platforms, platformthemes, styles - QtGui
        # mediaservice - QtMultimedia
        # printsupport - QtPrintSupport
        for plugin_name in (
                # "accessible",
                # "iconengines",
                # "platforminputcontexts",
                # "xcbglintegrations",
                # "egldeviceintegrations",
                "wayland-decoration-client",
                "wayland-graphics-integration-client",
                # "wayland-graphics-integration-server",
                "wayland-shell-integration",
        ):
            include_files += get_qt_plugins_paths("PySide6", plugin_name)
    # base="Win32GUI" should be used only for Windows GUI app
    base = "Win32GUI" if sys.platform == "win32" else None

    build_exe_options = {
        "bin_excludes": ["libqpdf.so", "libqpdf.dylib"],
        # exclude packages that are not really needed
        "excludes": ["tkinter"],
        "packages": ["qutip", "sklearn", "markdown", "pyvisa_py"],
        "include_files": include_files,
        "zip_include_packages": ["PySide6"],
    }

    # NO GUI
    executables = [Executable("visage.py", base=base, targetName="visage.exe", icon="./package_data/favicon.ico")]

    # Has GUI
    # executables = [Executable("visage.py", targetName="visage.exe", icon="./package_data/favicon.ico")]

    setup(
        name="PyQCat-Visage",
        version=VERSION,
        description=DESCRIPTION,
        options={"build_exe": build_exe_options},
        executables=executables,
    )
    try:
        import sklearn

        env_path = None
        for path_ in sys.path:
            if path_.endswith("site-packages") and "lib" in path_:
                env_path = path_
        if env_path:
            compile_path = project_path.joinpath("build").joinpath(
                f"exe.win-amd64-{'.'.join([str(x) for x in sys.version_info[:2]])}")
            lib_path = compile_path.joinpath("lib")
            sklearn_path = Path(env_path).joinpath("sklearn").joinpath(".libs")
            target_sk_path = lib_path.joinpath("sklearn").joinpath(".libs")
            for file in sklearn_path.rglob("*.dll"):
                if file.is_file():
                    shutil.copy(file, target_sk_path.joinpath(file.name))
            # delete project code
            project_names = ["pyQCat", "pyqcat_visage"]
            for name in project_names:
                remove_path = lib_path.joinpath(name)
                shutil.rmtree(remove_path)
            compile_path.rename(project_path.joinpath("build").joinpath(datetime.now().strftime("%Y_%m%d_%H%M")))
        else:
            print(f"Not find env path from {sys.path}")
    except:
        pass

elif len(sys.argv) > 2 and sys.argv[2] == "bdist_nuitka " or \
        len(sys.argv) == 2 and sys.argv[1] == "bdist_nuitka":

    # --windows-disable-console
    os.system("nuitka --mingw64 --standalone --windows-disable-console --show-progress --show-memory "
              "--enable-plugin=pyside6 --nofollow-import-to=numpy,pandas,zmq,cffi,tkinter,"
              "sympy,scipy,minio,yaml,sklearn,jsonschema,pyvisa,lmfit,qutip,loguru,rich,tqdm,networkx,markdown,"
              "pdfkit,requests,pyrsistent,jinja2,urllib3,zopfli,qtpy,qt_material,pytz,PIL,idna,kiwisolver,"
              "dateutil,pyparsing,packaging,email,http,attr,qdarkstyle,typing_extensions,pymongo,mongoengine,"
              "*.tests,*.example,*.examples,*.__pycache__ --include-module=PySide6.QtOpenGL,bson "
              "--windows-icon-from-ico=./package_data/favicon.ico "
              "--output-dir=build ./visage.py")
