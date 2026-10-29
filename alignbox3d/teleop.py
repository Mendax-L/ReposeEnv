"""AlignBox3D human control and Python collection entry point."""
from common.controllers import SpatialTeleop


def create_teleop(environment):
    return SpatialTeleop(environment)


def main():
    from common.recording import main as record_main
    record_main(default_env="alignbox3d")


if __name__ == "__main__":
    main()
