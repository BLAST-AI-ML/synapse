import subprocess


def run(cmd: str, proceed: bool = False):
    if proceed in ["y"]:
        try:
            subprocess.run(
                cmd,
                check=True,
                encoding="utf-8",
                shell=True,
                text=True,
            )
        except Exception as msg:
            print(msg)
    else:
        print("Exiting...")


if __name__ == "__main__":
    # prune all existing images
    proceed = input("\nPrune all existing images? [y/N] ")
    cmd = "docker system prune -a -f"
    run(cmd, proceed)

    # build the new image
    proceed = input("\nBuild new image? [y/N] ")
    cmd = "docker build --platform linux/amd64 -t gui -f dashboard/Dockerfile ."
    run(cmd, proceed)

    # upload to the NERSC registry
    proceed = input("\nPublish new image? [y/N] ")
    cmd_list = [
        "docker login registry.nersc.gov",
        "docker tag gui:latest registry.nersc.gov/m558/superfacility/gui:latest",
        "docker tag gui:latest registry.nersc.gov/m558/superfacility/gui:$(date '+%y.%m')",
        "docker push -a registry.nersc.gov/m558/superfacility/gui",
    ]
    cmd = " && ".join(cmd_list)
    run(cmd, proceed)
