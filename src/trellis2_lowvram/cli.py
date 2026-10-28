"""Command line entry point: `trellis2-lowvram <command>`.

User-facing text goes through i18n.t() and is printed; diagnostics go to the log file.
"""

from __future__ import annotations

import argparse
import errno
import getpass
import json
import logging
import sys
import time
from pathlib import Path

from . import __version__
from .config import REPO_ROOT, prepare_environment
from .i18n import set_lang, t

log = logging.getLogger("trellis2_lowvram")

MARK = {"ok": "[ OK ]", "warn": "[ !! ]", "fail": "[FAIL]", "skip": "[ -- ]"}


def setup_logging(logs_dir: Path, verbose: bool) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"trellis2-lowvram-{time.strftime('%Y%m%d-%H%M%S')}.log"
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(fh)
    if verbose:
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(logging.DEBUG)
        root.addHandler(sh)
    return path


def say(text: str = "") -> None:
    print(text, flush=True)


# ---------------------------------------------------------------- doctor


def cmd_doctor(args: argparse.Namespace, paths) -> int:
    from . import env_check
    from .env_check import Status

    say(t("doctor.title", version=__version__))
    checks = env_check.run_hardware_checks(REPO_ROOT, paths.models)
    if not args.pre:
        checks += env_check.check_software()
        checks.append(_access_check())

    for c in checks:
        say(f"{MARK[c.status.value]} {t(c.msg, **c.params)}")
        if c.fix and c.status in (Status.WARN, Status.FAIL):
            say(f"       -> {t(c.fix, **c.params)}")
        log.info("check %s %s %s", c.id, c.status.value, c.params)

    result = env_check.worst(checks)
    say()
    say(t(f"doctor.result.{result.value}"))
    if args.json:
        report = {
            "version": __version__,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "system": env_check.system_summary(),
            "result": result.value,
            "checks": [c.to_dict() for c in checks],
        }
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
        say(t("doctor.saved", path=args.json))
    return 1 if result == Status.FAIL else 0


def _access_check():
    from .env_check import Check, Status
    from .hf_access import DINOV3_REPO, DINOV3_URL, Access, check_access

    a = check_access()
    p = {"repo": DINOV3_REPO, "url": DINOV3_URL}
    if a == Access.OK:
        return Check("hf_access", Status.OK, "check.hf.ok", params=p)
    if a == Access.NETWORK:
        return Check("hf_access", Status.WARN, "check.hf.network", "fix.hf.network", p)
    return Check("hf_access", Status.FAIL, f"check.hf.{a.value}", "fix.hf.login", p)


# ---------------------------------------------------------------- login


def cmd_login(args: argparse.Namespace, paths) -> int:
    from .hf_access import DINOV3_URL, JOIN_URL, TOKEN_URL, Access, check_access, save_token

    say(t("login.intro"))
    say(t("login.step1", url=JOIN_URL))
    say(t("login.step2", url=DINOV3_URL))
    say(t("login.step3", url=TOKEN_URL))
    say()
    token = (args.token or getpass.getpass(t("login.prompt"))).strip()
    if not token:
        say(t("login.empty"))
        return 1
    a = check_access(token=token)
    if a in (Access.BAD_TOKEN, Access.NO_TOKEN):
        say(t("check.hf.bad_token"))
        return 1
    if a == Access.NETWORK:
        say(t("check.hf.network"))
        return 1
    save_token(token)
    say(t("login.saved"))
    if a == Access.NOT_APPROVED:
        say(t("check.hf.not_approved", url=DINOV3_URL))
        say(t("fix.hf.wait"))
        return 2
    say(t("check.hf.ok"))
    return 0


# ---------------------------------------------------------------- download / extensions


def cmd_download(args: argparse.Namespace, paths) -> int:
    from .hf_access import DINOV3_URL, Access, check_access
    from .models import download

    if check_access() != Access.OK:
        say(t("download.need_access", url=DINOV3_URL))
        return 2
    say(t("download.start", path=paths.models))
    try:
        download(tuple(args.pipeline))
    except OSError as e:
        log.exception("download failed")
        if e.errno == errno.ENOSPC:
            say(t("download.no_space", path=paths.models))
        else:  # requests' connection errors are OSError subclasses too
            say(t("download.failed"))
        return 1
    except Exception:
        log.exception("download failed")
        say(t("download.failed"))
        return 1
    say(t("download.done"))
    return 0


def cmd_install_extensions(args: argparse.Namespace, paths) -> int:
    from .extensions import entries_for, install, load_manifest, platform_key

    if not entries_for(load_manifest(), platform_key()):
        say(t("ext.not_ready"))
        return 1
    names = install(paths.home / ".tools" / "wheels")
    say(t("ext.installed", names=", ".join(names)))
    return 0


# ---------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    from .models import PIPELINE_TYPES

    p = argparse.ArgumentParser(prog="trellis2-lowvram")
    p.add_argument("--lang", choices=["ko", "en"], help="message language (default: OS)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="check this PC")
    d.add_argument("--pre", action="store_true", help="hardware only (before install)")
    d.add_argument("--json", metavar="PATH", help="also save a JSON report")
    d.set_defaults(func=cmd_doctor)

    lg = sub.add_parser("login", help="save a Hugging Face token and check DINOv3 access")
    lg.add_argument("--token", help="token (default: ask)")
    lg.set_defaults(func=cmd_login)

    dl = sub.add_parser("download", help="download model weights")
    dl.add_argument("--pipeline", nargs="+", choices=PIPELINE_TYPES, default=["512"])
    dl.set_defaults(func=cmd_download)

    ie = sub.add_parser("install-extensions", help="install prebuilt CUDA extension wheels")
    ie.set_defaults(func=cmd_install_extensions)
    return p


def main(argv: list[str] | None = None) -> int:
    paths = prepare_environment()
    args = build_parser().parse_args(argv)
    set_lang(args.lang)
    log_path = setup_logging(paths.logs, args.verbose)
    log.info("trellis2-lowvram %s argv=%s", __version__, sys.argv)
    try:
        return args.func(args, paths)
    except KeyboardInterrupt:
        return 130
    except Exception:
        log.exception("unhandled error")
        say(t("error.unexpected", log=log_path))
        return 1


if __name__ == "__main__":
    sys.exit(main())
