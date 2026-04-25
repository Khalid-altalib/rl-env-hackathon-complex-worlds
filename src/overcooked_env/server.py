from openreward.environments import Server

from .env import OvercookedEnv


def main():
    Server([OvercookedEnv]).run()


if __name__ == "__main__":
    main()
