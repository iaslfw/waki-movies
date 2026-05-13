import subprocess


def main():
    print("Hello from waki-movies!")
    subprocess.run(["uvx", "pycowsay", "WakiKaki"])


if __name__ == "__main__":
    main()
