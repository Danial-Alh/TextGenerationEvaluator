import os
import re
import sys

root_path = './previous_works/{}'.format(sys.argv[1])

print(root_path)
input('continue?')

def refactor_file_imports(path, import_level):
    with open(path, 'r', encoding='utf8') as file:
        lines = file.readlines()
    lines = [l.replace('\n', '') for l in lines]
    for i, line in enumerate(lines):
        if 'import ' in line:
            if 'from' in line:
                after_from = line[line.find("from ") + len("from "):]
                for prefix in root_folders:
                    if after_from.startswith(prefix):
                        lines[i] = line.replace('from ', 'from %s ' % import_level)
            else:
                after_import = line[line.find("import ") + len("import "):]
                for prefix in root_folders:
                    if after_import.startswith(prefix):
                        lines[i] = line.replace('import', 'from %s import' % import_level)
        elif re.match(r'^\s*\sprint\s+', line) is not None:
            lines[i] = re.sub(r'^\s*\sprint\s+', r'?1', line) + ')'
            lines[i] = re.sub(r'\sprint\s+', r'\g<0>(', line) + ')'
        elif 'xrange' in line:
            lines[i] = re.sub('xrange\s*\(', 'range(', line)
    with open(path, 'w', encoding='utf8') as file:
        file.write("\n".join(lines))
    print('%s refactored' % (path))


def refactor_folder_imports(curr_path=root_path, import_level='.'):
    for file in os.listdir(curr_path):
        path = curr_path + '/' + file
        if os.path.isdir(path):
            refactor_folder_imports(path, import_level + '.')
        elif path.endswith('.py'):
            refactor_file_imports(path, import_level)


root_folders = []


def extract_root_folders(p=root_path):
    for file in os.listdir(p):
        path = p + '/' + file
        if os.path.isdir(path):
            root_folders.append(file)
        if file.endswith('.py'):
            root_folders.append(file[:-3])


extract_root_folders()
refactor_folder_imports()
