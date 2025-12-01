import os
def set_cwd():
    project_root = os.getenv("PYTHONPATH")
    os.chdir(project_root)