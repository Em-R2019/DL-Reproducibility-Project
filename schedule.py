import os
import subprocess

if __name__ == "__main__":
    config_path = "configs"
    configs = os.listdir(config_path)
    configs.sort()
    for config in configs:
        name = os.path.join(config_path, config)
        newname = os.path.join(config_path, "config.yaml")
        os.replace(name, newname)
        print(name, newname)
        print(os.getcwd())

        subprocess.Popen(["venv/Scripts/python.exe", "test.py"]).wait()
        print('Finished testing ' + config)
        # subprocess.run('python test.py')
        # print('Finished testing ' + config)
        os.rename(newname, name)
