import subprocess as subp
from subprocess import Popen

from config import Config


class Utils(object):

    def __init__(self) -> None:
        self.cfg = Config()
        self.os_name = self.cfg.os_name
        self.kill_proc_cmd_win = [
            "taskkill",
            "/F",
            "/T",
            "/PID"
        ]
        self.kill_proc_cmd_linux_and_macos = ["kill", "-9"]

    def _run_kill_cmd(
        self,
        cmd: list[str],
        pid: int
    ) -> str:
        with Popen(
            cmd + [str(pid)],
            stdout=subp.PIPE,
            stderr=subp.PIPE
        ) as proc:
            _, proc_err = proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"Failed to terminated process: {pid}\n{proc_err.decode('utf-8')}")

        return ''

    def kill_proc(
        self,
        pid: int
    ) -> str:
        if self.os_name == "win32":
            return self._run_kill_cmd(self.kill_proc_cmd_win, pid)
        elif self.os_name == "linux" or self.os_name == "darwin":
            return self._run_kill_cmd(self.kill_proc_cmd_linux_and_macos, pid)
        else:
            raise OSError(f"Unsupported OS: {self.os_name}")
