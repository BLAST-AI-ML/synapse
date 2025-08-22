import subprocess


def run(command: str, proceed: bool = False):
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


if __name__ == "__main__":
    # prune all existing images
    proceed = input("\nPrune all existing images? [y/N] ")
    command = "docker system prune -a -f"
    run(command, proceed)

    # build the new image
    proceed = input("\nBuild new image? [y/N] ")
    command = "docker build --platform linux/amd64 -t gui -f dashboard/Dockerfile ."
    run(command, proceed)

    # upload to the NERSC registry
    proceed = input("\nPublish new image? [y/N] ")
    command_list = [
        "docker login registry.nersc.gov",
        "docker tag gui:latest registry.nersc.gov/m558/superfacility/gui:latest",
        "docker tag gui:latest registry.nersc.gov/m558/superfacility/gui:$(date '+%y.%m')",
        "docker push -a registry.nersc.gov/m558/superfacility/gui",
    ]
    command = " && ".join(command_list)
    run(command, proceed)
