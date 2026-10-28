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
        checks.append(_models_check())

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
    fix = "fix.hf.token_no_gated" if a == Access.TOKEN_NO_GATED else "fix.hf.login"
    return Check("hf_access", Status.FAIL, f"check.hf.{a.value}", fix, p)


def _models_check(pipeline_type: str = "512"):
    from .env_check import Check, Status
    from .models import missing_files

    missing = missing_files(pipeline_type)
    if not missing:
        return Check("models", Status.OK, "check.models.ok")
    fix = "fix.models.missing" if pipeline_type == "512" else "fix.models.missing_1024"
    return Check(
        "models",
        Status.FAIL,
        "check.models.missing",
        fix,
        {"n": len(missing), "files": missing},
    )


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


# ---------------------------------------------------------------- generate / benchmark


def _prepare_gpu() -> None:
    from . import attention

    attention.apply(attention.choose_backend())


def _ready_to_generate(pipeline_type: str) -> bool:
    """Check what we can before the 1-2 minute model load, and say what to do if it fails."""
    import torch

    from .env_check import Status

    if not torch.cuda.is_available():
        say(t("error.no_gpu"))
        say(f"       -> {t('fix.torch.no_cuda')}")
        return False
    c = _models_check(pipeline_type)
    if c.status != Status.OK:
        say(t(c.msg, **c.params))
        say(f"       -> {t(c.fix, **c.params)}")
        return False
    return True


def _readable(images: list[Path]) -> list[Path]:
    from .images import BadImage, open_image

    ok = []
    for p in images:
        if not p.is_file():
            say(t("gen.no_input", path=p))
            continue
        try:
            open_image(p)
        except BadImage:
            say(t("error.bad_image", path=p.name))
            say(f"       -> {t('fix.bad_image')}")
            continue
        ok.append(p)
    return ok


def _ask_retry(interactive: bool):
    def ask(stage: str, cur, new) -> bool:
        say(t("gen.oom", stage=stage))
        say(t("gen.oom.proposal", old=_describe(cur), new=_describe(new)))
        if not interactive:
            say(t("gen.oom.auto"))
            return True
        return input(t("gen.oom.ask")).strip().lower() in ("y", "yes", "예", "ㅇ")

    return ask


def _describe(s) -> str:
    return f"{s.pipeline_type} / steps {s.steps} / texture {s.texture_size}"


def _job_from_args(args: argparse.Namespace, image: Path, out_dir: Path):
    from .generate import Job
    from .presets import resolve

    settings = resolve(
        args.preset,
        pipeline_type=args.pipeline,
        texture_size=args.texture_size,
        steps=args.steps,
    )
    seeds = [args.seed + i for i in range(args.seeds)]
    return Job(image, out_dir, settings, seeds, args.vram_cap, obj=not args.no_obj)


def cmd_generate(args: argparse.Namespace, paths) -> int:
    from .generate import run_job, write_report
    from .presets import resolve

    images = _readable([Path(p) for p in args.input])
    if not images:
        return 1
    pipeline_type = resolve(args.preset, pipeline_type=args.pipeline).pipeline_type
    if not _ready_to_generate(pipeline_type):
        return 1
    _prepare_gpu()
    from .memory import StageMonitor
    from .pipeline import Generator

    # A new folder per run, like the web UI: dropping the same picture again must not overwrite
    # the earlier result, and --open shows exactly what this run made.
    out_dir = Path(args.output) if args.output else paths.outputs / time.strftime("%Y%m%d-%H%M%S")
    ask = _ask_retry(sys.stdin.isatty() and not args.yes)
    gen, n_ok = None, 0
    for image in images:
        job = _job_from_args(args, image, out_dir)
        say(t("gen.start", image=image.name, seeds=len(job.seeds), preset=args.preset))
        if gen is None:  # load once, reuse for every image
            say(t("gen.loading"))
            gen = Generator(
                job.settings.pipeline_type, StageMonitor(on_stage=_say_stage), on_load=_say_load
            )
        report = run_job(job, ask, gen=gen)
        jp, _ = write_report(report, out_dir, f"{image.stem}-report")
        for r in report["results"]:
            if r["ok"]:
                n_ok += 1
                say(t("gen.done", seed=r["seed"], path=r["files"][0], seconds=r["seconds"]))
        say(t("gen.report", path=jp))
    if n_ok:
        say()
        say(t("gen.saved_in", path=out_dir))
        if args.open:
            _open_folder(out_dir)
    return 0 if n_ok else 1


def _open_folder(folder: Path) -> None:
    import os
    import subprocess

    # os.startfile has failed with "server execution failed" (seen in a remote session); try
    # explorer before giving up. The folder path is printed either way (gen.saved_in).
    if os.name == "nt":
        commands = [None, ["explorer", str(folder)]]
    else:
        commands = [["xdg-open", str(folder)]]
    for cmd in commands:
        try:
            if cmd is None:
                os.startfile(folder)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(cmd)
            return
        except Exception:
            log.debug("could not open %s with %s", folder, cmd or "startfile", exc_info=True)


def cmd_benchmark(args: argparse.Namespace, paths) -> int:
    from .config import UPSTREAM_DIR
    from .generate import run_job, write_report
    from .presets import resolve

    if not _ready_to_generate(resolve(args.preset, pipeline_type=args.pipeline).pipeline_type):
        return 1
    _prepare_gpu()
    image = Path(args.input) if args.input else UPSTREAM_DIR / "assets" / "example_image" / "T.png"
    out_dir = (
        Path(args.out) if args.out else paths.logs / f"benchmark-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    job = _job_from_args(args, image, out_dir)
    report = run_job(job, None, on_stage=_say_stage)
    jp, mp = write_report(report, out_dir, "benchmark")
    say(mp.read_text(encoding="utf-8"))
    say(t("gen.report", path=jp))
    return 0 if all(r["ok"] for r in report["results"]) else 1


def cmd_ui(args: argparse.Namespace, paths) -> int:
    import torch

    if not torch.cuda.is_available():  # the page itself works; generation will not
        say(t("error.no_gpu"))
        say(f"       -> {t('fix.torch.no_cuda')}")
    _prepare_gpu()
    from .ui.app import launch

    if args.share:
        say(t("ui.share_warning"))
    launch(share=args.share, port=args.port, lang=args.lang, browser=not args.no_browser)
    return 0


def _say_load(done: int, total: int, part: str) -> None:
    if part != "done":
        say(t("load.progress", done=done + 1, total=total, part=t(f"load.{part}")))


def _say_stage(stage: str) -> None:
    say(t("gen.stage", stage=t(f"stage.{stage}")))


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

    def add_gen_options(sp: argparse.ArgumentParser) -> None:
        from .presets import PRESETS

        sp.add_argument("--preset", choices=list(PRESETS), default="balanced")
        sp.add_argument("--pipeline", choices=PIPELINE_TYPES, help="512 (default) or 1024_cascade")
        sp.add_argument("--texture-size", type=int, choices=[1024, 2048, 4096])
        sp.add_argument("--steps", type=int, help="sampling steps for every stage")
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--seeds", type=int, default=1, help="how many seeds, from --seed up")
        sp.add_argument("--vram-cap", type=float, metavar="GB", help="simulate a smaller GPU")
        sp.add_argument("--no-obj", action="store_true", help="GLB only")

    g = sub.add_parser("generate", help="image -> 3D model (GLB, OBJ)")
    g.add_argument(
        "input",
        nargs="+",
        help="image file(s) (PNG/JPG/WEBP); transparent PNG skips background removal",
    )
    g.add_argument("--open", action="store_true", help="open the output folder when done")
    g.add_argument("-o", "--output", help="output folder (default: outputs/)")
    g.add_argument("-y", "--yes", action="store_true", help="accept lower settings after OOM")
    add_gen_options(g)
    g.set_defaults(func=cmd_generate)

    b = sub.add_parser("benchmark", help="measure VRAM and time per stage")
    b.add_argument("--input", help="image (default: upstream sample T.png)")
    b.add_argument("--out", help="report folder (default: logs/benchmark-<time>)")
    add_gen_options(b)
    b.set_defaults(func=cmd_benchmark)

    u = sub.add_parser("ui", help="open the web interface (local only)")
    u.add_argument("--port", type=int, help="default: first free port from 7860")
    u.add_argument("--share", action="store_true", help="also make a public gradio.live link")
    u.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    u.set_defaults(func=cmd_ui)
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
    except Exception as e:
        from .errors import explain

        log.exception("unhandled error")
        x = explain(e)
        if x is None:
            say(t("error.unexpected", log=log_path))
        else:
            say(t(x.msg, **x.params))
            say(f"       -> {t(x.fix, **x.params)}")
            say(t("error.log", log=log_path))
        return 1


if __name__ == "__main__":
    sys.exit(main())
