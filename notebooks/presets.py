import os


def set_cwd(subdir: str = ""):
    project_root = os.getenv("PYTHONPATH")
    os.chdir(os.path.join(project_root, subdir))
