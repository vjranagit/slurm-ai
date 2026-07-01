from __future__ import annotations

import argparse
import os
import tempfile

from controller.slurm_exec import SlurmCommandRunner, SlurmExecConfig


def submit_jobs(
    compose_file: str,
    service: str,
    count: int,
    seconds: int,
    exec_mode: str = "docker",
    ssh_host: str = "",
    ssh_user: str = "",
    ssh_key_file: str = "",
) -> None:
    runner = SlurmCommandRunner(
        SlurmExecConfig(
            mode=exec_mode,
            compose_file=compose_file,
            service=service,
            ssh_host=ssh_host,
            ssh_user=ssh_user,
            ssh_key_file=ssh_key_file,
        )
    )
    for i in range(count):
        fd, script = tempfile.mkstemp(prefix="adaptive-job-", suffix=".sh", dir="/tmp")
        os.close(fd)
        create_script = (
            "cat > {path} <<'EOS'\n"
            "#!/bin/bash\n"
            "set -euo pipefail\n"
            "sleep {seconds}\n"
            "echo job-{i}-done\n"
            "EOS\n"
            "chmod +x {path}"
        ).format(path=script, seconds=seconds, i=i)
        try:
            ok_create, out_create = runner.run(create_script, check=True)
            if not ok_create:
                raise RuntimeError(out_create)

            ok_submit, out_submit = runner.run(f"sbatch {script}", check=True)
            if not ok_submit:
                raise RuntimeError(out_submit)
            print(out_submit.strip())
        finally:
            # `script` is a local placeholder created only to get an unpredictable
            # name (the real script is written remotely via the heredoc above).
            # Remove it so we don't leak a 0-byte file per submitted job; in local
            # mode the heredoc/sbatch may already have consumed or replaced it.
            try:
                os.unlink(script)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose-file", default="infra/docker/docker-compose.yml")
    parser.add_argument("--service", default="slurm")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seconds", type=int, default=5)
    parser.add_argument("--exec-mode", default="docker")
    parser.add_argument("--ssh-host", default="")
    parser.add_argument("--ssh-user", default="")
    parser.add_argument("--ssh-key-file", default="")
    args = parser.parse_args()
    submit_jobs(
        args.compose_file,
        args.service,
        args.count,
        args.seconds,
        exec_mode=args.exec_mode,
        ssh_host=args.ssh_host,
        ssh_user=args.ssh_user,
        ssh_key_file=args.ssh_key_file,
    )
