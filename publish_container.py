#!/usr/bin/env python
"""
Build the Dashboard (GUI) container or
build the ML training container (with CUDA support) for
Perlmutter (NERSC) and publish it to registry.nersc.gov
"""

import argparse
import subprocess


def run(command: str, proceed: str, auto_yes: bool):
    if auto_yes or proceed in ["y"]:
        try:
            subprocess.run(
                command,
                check=True,
                encoding="utf-8",
                shell=True,
                text=True,
            )
        except Exception as error:
            print(error)
    else:
        print("Skipping...")


def build_container(container: str, auto_yes: bool):
    # where to find the Dockerfile
    folders = {
        "gui": "dashboard",
        "ml": "ml",
    }
    # how to name the container image
    imagename = {
        "gui": "gui",
        "ml": "ml-training",
    }

    # build the new image
    proceed = "y" if auto_yes else input(f"\nBuild new {container} image? [y/N] ")
    command = f"docker build --platform linux/amd64 -t {imagename[container]} -f {folders[container]}.Dockerfile ."
    run(command, proceed, auto_yes)

    # upload to the NERSC registry
    proceed = "y" if auto_yes else input(f"\nPublish new {container} image? [y/N] ")
    command_list = [
        "docker login registry.nersc.gov",
        f"docker tag {imagename[container]}:latest registry.nersc.gov/m558/superfacility/{imagename[container]}:latest",
        f"docker tag {imagename[container]}:latest registry.nersc.gov/m558/superfacility/{imagename[container]}:$(date '+%y.%m')",
        f"docker push -a registry.nersc.gov/m558/superfacility/{imagename[container]}",
    ]
    command = " && ".join(command_list)
    run(command, proceed, auto_yes)


if __name__ == "__main__":
    # CLI options: --gui --ml
    parser = argparse.ArgumentParser(description="Build a container image and push it to the NERSC registry.")
    parser.add_argument(
        "--gui", action="store_true", help="Build Dashboard GUI container"
    )
    parser.add_argument("--ml", action="store_true", help="Build ML training container")
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Automatically answer yes to all prompts",
    )
    args = parser.parse_args()

    containers = []
    containers += ["gui"] if args.gui else []
    containers += ["ml"] if args.ml else []
    if len(containers) == 0:
        parser.error("At least one of the options --gui or --ml must be set.")

    # prune all existing images
    proceed = "y" if args.yes else input("\nPrune all existing images? [y/N] ")
    command = "docker system prune -a -f"
    run(command, proceed, args.yes)

    # build new container images
    for container in containers:
        build_container(container, args.yes)
