from openreward.environments import Server

from .env import CatanEnv


def main():
    Server([CatanEnv]).run()


if __name__ == "__main__":
    main()
