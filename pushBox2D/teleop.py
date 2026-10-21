"""PushBox2D human control and Python collection entry point."""
from common.controllers import PlanarTeleop


def create_teleop(environment):
    return PlanarTeleop(environment)


def main():
    from record import main as record_main
    record_main(default_env="pushbox2d")


if __name__ == "__main__":
    main()
