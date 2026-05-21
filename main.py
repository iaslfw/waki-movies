import subprocess
from src.data.prepare_data import prepare_local_data


def main():
    print("Hello from waki-movies!")
    subprocess.run(["uvx", "pycowsay", "WakiKaki"])

    prepare_local_data()


if __name__ == "__main__":
    main()
