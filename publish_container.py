#!/usr/bin/env python
"""
Build the Dashboard (GUI) container or
build the ML training container (with CUDA support) for
Perlmutter (NERSC) and publish it to registry.nersc.gov
"""
import argparse
import subprocess


def run(command: str, proceed: str):
    if proceed in ["y"]:
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


def build_container(container: str):
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
    proceed = input(f"\nBuild new {container} image? [y/N] ")
    command = f"docker build --platform linux/amd64 -t {imagename[container]} -f {folders[container]}.Dockerfile ."
    run(command, proceed)

    # upload to the NERSC registry
    proceed = input("\nPublish new image? [y/N] ")
    command_list = [
        "docker login registry.nersc.gov",
        f"docker tag gui:latest registry.nersc.gov/m558/superfacility/{imagename[container]}:latest",
        f"docker tag gui:latest registry.nersc.gov/m558/superfacility/{imagename[container]}:$(date '+%y.%m')",
        f"docker push -a registry.nersc.gov/m558/superfacility/{imagename[container]}",
    ]
    command = " && ".join(command_list)
    run(command, proceed)


if __name__ == "__main__":
    # CLI options: --gui --ml
    parser = argparse.ArgumentParser(description="Containerts to build:")
    parser.add_argument('--gui', action='store_true', help='Build Dashboard GUI container')
    parser.add_argument('--ml', action='store_true', help='Build ML training container')
    args = parser.parse_args()
    
    containers = []
    containers += ["gui"] if args.gui else []
    containers += ["ml"] if args.ml else []
    if len(containers) == 0:
        parser.error("At least one of the options --gui or --ml must be set.")

    # prune all existing images
    proceed = input("\nPrune all existing images? [y/N] ")
    command = "docker system prune -a -f"
    run(command, proceed)

    # build new container images
    for container in containers:
        build_container(container)

