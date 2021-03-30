from config import Config

from agent.ingan_agent import InganAgent


def main():
    config = Config()

    agent = InganAgent(config)
    agent.run()


if __name__ == '__main__':
    main()
