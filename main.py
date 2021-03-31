from config import Config

from agent.discriminator_agent import DiscriminatorAgent


def main():
    config = Config()

    agent = DiscriminatorAgent(config)
    agent.run()


if __name__ == '__main__':
    main()
