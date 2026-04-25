from openreward.environments import Server

from .env import DiplomacyEnv


def main():
    Server([DiplomacyEnv]).run()


if __name__ == "__main__":
    main()
