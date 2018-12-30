def compile_cython():
    import os, inspect
    ROOT_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/../'
    os.system("python {}setup.py build_ext --build-lib {}/lib".format(ROOT_PATH, ROOT_PATH))


compile_cython()
