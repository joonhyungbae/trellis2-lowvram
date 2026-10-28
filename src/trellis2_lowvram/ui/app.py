"""Gradio web UI: upload -> (background removal) -> generate seeds -> 3D preview -> download.

Runs locally on 127.0.0.1. `share=True` (a public gradio.live link) only when the user asks for it.
The pipeline is loaded once, on the first generation, and reused; one generation runs at a time.
"""

from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import gradio as gr

from .. import i18n
from ..config import get_paths
from ..presets import PRESETS, Settings, lower

log = logging.getLogger(__name__)

MAX_SEEDS = 4
PRESET_KEYS = ("fast", "balanced", "quality")


def tr(lang: str, key: str, **kw: Any) -> str:
    i18n.set_lang(lang)
    return i18n.t(key, **kw)


class Runner:
    """Holds the loaded pipeline between requests."""

    def __init__(self) -> None:
        self.gen = None
        self.lock = threading.Lock()

    def generator(self, pipeline_type: str):
        from ..memory import StageMonitor
        from ..pipeline import Generator

        if self.gen is None:
            self.gen = Generator(pipeline_type, StageMonitor())
        return self.gen


def vram_line(lang: str) -> str:
    import torch

    free, total = torch.cuda.mem_get_info()
    return tr(
        lang, "ui.vram", used=f"{(total - free) / 1024**3:.1f}", total=f"{total / 1024**3:.1f}"
    )


def build(runner: Runner, default_lang: str) -> gr.Blocks:
    import torch

    gpu = torch.cuda.get_device_name(0)
    labels: list[tuple[Any, str, str]] = []  # (component, i18n key, attribute) for language switch

    def L(component, key: str, attr: str = "label"):
        labels.append((component, key, attr))
        return component

    t = lambda key, **kw: tr(default_lang, key, **kw)  # noqa: E731

    with gr.Blocks(title="trellis2-lowvram") as demo:
        with gr.Row():
            L(gr.Markdown(t("ui.title", gpu=gpu)), "ui.title", "value")
            lang = gr.Radio(
                [("한국어", "ko"), ("English", "en")],
                value=default_lang,
                show_label=False,
                scale=0,
                min_width=200,
            )
        L(gr.Markdown(t("ui.notice")), "ui.notice", "value")
        with gr.Row():
            with gr.Column(scale=1):
                image = L(gr.Image(type="filepath", label=t("ui.image"), height=320), "ui.image")
                remove_bg = L(
                    gr.Checkbox(value=True, label=t("ui.remove_bg"), info=t("ui.remove_bg.info")),
                    "ui.remove_bg",
                )
                preset = L(
                    gr.Radio(
                        [(t(f"ui.preset.{k}"), k) for k in PRESET_KEYS],
                        value="balanced",
                        label=t("ui.preset"),
                    ),
                    "ui.preset",
                )
                with L(gr.Accordion(t("ui.advanced"), open=False), "ui.advanced"):
                    seed = L(gr.Number(value=42, precision=0, label=t("ui.seed")), "ui.seed")
                    n_seeds = L(
                        gr.Slider(
                            1,
                            MAX_SEEDS,
                            value=1,
                            step=1,
                            label=t("ui.n_seeds"),
                            info=t("ui.n_seeds.info"),
                        ),
                        "ui.n_seeds",
                    )
                    resolution = L(
                        gr.Radio(
                            [("512", "512"), ("1024 (experimental)", "1024_cascade")],
                            value="512",
                            label=t("ui.resolution"),
                            info=t("ui.resolution.info"),
                        ),
                        "ui.resolution",
                    )
                    texture = L(
                        gr.Radio([1024, 2048, 4096], value=2048, label=t("ui.texture")),
                        "ui.texture",
                    )
                    tex_guidance = L(
                        gr.Slider(
                            1.0,
                            10.0,
                            value=1.0,
                            step=0.1,
                            label=t("ui.tex_guidance"),
                            info=t("ui.tex_guidance.info"),
                        ),
                        "ui.tex_guidance",
                    )
                    faces = L(
                        gr.Slider(
                            100_000, 1_000_000, value=500_000, step=50_000, label=t("ui.faces")
                        ),
                        "ui.faces",
                    )
                    steps = gr.State(12)
                run = L(gr.Button(t("ui.run"), variant="primary"), "ui.run", "value")
                status = gr.Markdown(vram_line(default_lang))
            with gr.Column(scale=2):
                with gr.Row():
                    views = [
                        gr.Model3D(
                            label="seed",
                            height=360,
                            visible=(i == 0),
                            clear_color=(0.93, 0.93, 0.93, 1.0),
                        )
                        for i in range(MAX_SEEDS)
                    ]
                previews = L(
                    gr.Gallery(
                        label=t("ui.previews"), columns=1, height=280, object_fit="contain"
                    ),
                    "ui.previews",
                )
                files = L(gr.File(label=t("ui.files"), file_count="multiple"), "ui.files")
                with L(gr.Accordion(t("ui.report"), open=False), "ui.report"):
                    report = gr.Markdown()

        # ---------------------------------------------------------------- events

        def on_preset(p: str):
            s = PRESETS[p]
            return s.texture_size, s.decimation_target, s.steps

        preset.change(on_preset, preset, [texture, faces, steps])

        def on_lang(lg: str):
            out = []
            for comp, key, attr in labels:
                if isinstance(comp, gr.Radio) and key == "ui.preset":
                    out.append(
                        gr.update(
                            label=tr(lg, key),
                            choices=[(tr(lg, f"ui.preset.{k}"), k) for k in PRESET_KEYS],
                        )
                    )
                elif attr == "value":
                    kw = {"gpu": gpu} if key == "ui.title" else {}
                    out.append(gr.update(value=tr(lg, key, **kw)))
                else:
                    upd = {attr: tr(lg, key)}
                    info_key = f"{key}.info"
                    if info_key in i18n.load("en"):
                        upd["info"] = tr(lg, info_key)
                    out.append(gr.update(**upd))
            return out

        lang.change(on_lang, lang, [c for c, _, _ in labels])

        def on_run(img, rm_bg, p, sd, n, res, tex, guid, fc, st, lg):
            hidden = [gr.update(visible=(i == 0), value=None) for i in range(MAX_SEEDS)]
            if not img:
                yield (
                    tr(lg, "ui.need_image"),
                    *hidden,
                    None,
                    None,
                    "",
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                )
                return
            s = replace(
                PRESETS[p],
                pipeline_type=res,
                texture_size=int(tex),
                steps=int(st),
                tex_guidance=float(guid),
                decimation_target=int(fc),
            )
            seeds = [int(sd) + i for i in range(int(n))]
            out_dir = get_paths().outputs / time.strftime("%Y%m%d-%H%M%S")
            events: queue.Queue = queue.Queue()
            box: dict[str, Any] = {}

            def work():
                from PIL import Image

                from ..generate import make_one

                try:
                    with runner.lock:
                        events.put("loading")
                        gen = runner.generator(s.pipeline_type)
                        gen.monitor.on_stage = events.put
                        source = Image.open(img)
                        stem = Path(img).stem[:40]
                        box["results"] = []
                        for seed in seeds:
                            events.put(f"seed:{seed}")
                            box["results"].append(
                                make_one(
                                    gen,
                                    source,
                                    seed,
                                    s,
                                    out_dir / f"{stem}_seed{seed}",
                                    None,
                                    bool(rm_bg),
                                    obj=True,
                                )
                            )
                except Exception as e:  # shown to the user, details in the log
                    log.exception("generation failed")
                    box["error"] = e
                finally:
                    events.put(None)

            th = threading.Thread(target=work, daemon=True)
            th.start()
            t0, stage, seed_now = time.perf_counter(), "loading", seeds[0]
            while True:
                try:
                    ev = events.get(timeout=0.7)
                except queue.Empty:
                    ev = ""
                if ev is None:
                    break
                if ev.startswith("seed:"):
                    seed_now = int(ev[5:])
                elif ev:
                    stage = ev
                stage_txt = tr(lg, "ui.loading") if stage == "loading" else tr(lg, f"stage.{stage}")
                msg = (
                    tr(
                        lg,
                        "ui.progress",
                        seed=seed_now,
                        stage=stage_txt,
                        seconds=int(time.perf_counter() - t0),
                    )
                    + "  \n"
                    + vram_line(lg)
                )
                # status only; views, previews, files, report and the 3 settings stay as they are
                yield (msg, *[gr.skip()] * (MAX_SEEDS + 6))
            th.join()

            if "error" in box:
                yield (
                    tr(lg, "ui.error", log=get_paths().logs),
                    *hidden,
                    None,
                    None,
                    "",
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                )
                return
            from ..preview import render

            views_out, downloads, lines, proposal, sheets = [], [], [], None, []
            for attempts in box["results"]:
                last = attempts[-1]
                if last.ok:
                    glb = next(f for f in last.files if f.endswith(".glb"))
                    views_out.append(gr.update(value=glb, visible=True, label=f"seed {last.seed}"))
                    downloads.append(glb)
                    try:
                        sheet = render(glb, str(Path(glb).with_suffix(".jpg")), img)
                        sheets.append((sheet, f"seed {last.seed}"))
                    except Exception:  # the 3D view and downloads still work without it
                        log.exception("preview render failed")
                    obj_dir = Path(glb).with_suffix("").parent / (Path(glb).stem + "_obj")
                    if obj_dir.is_dir():
                        downloads.append(shutil.make_archive(str(obj_dir), "zip", obj_dir))
                else:
                    proposal = proposal or lower(Settings(**last.settings), last.oom_stage)
                    lines.append(
                        tr(lg, "ui.oom", seed=last.seed, stage=tr(lg, f"stage.{last.oom_stage}"))
                    )
                for a in attempts:
                    lines.append(
                        f"- seed {a.seed}: {'ok' if a.ok else 'OOM ' + str(a.oom_stage)}, "
                        f"{a.seconds}s, peak "
                        f"{max((x['peak_reserved_mib'] for x in a.stages), default=0)} MiB"
                    )
            n_ok = len(views_out)
            views_out += [gr.update(value=None, visible=False)] * (MAX_SEEDS - n_ok)
            if n_ok == 0:
                views_out[0] = gr.update(value=None, visible=True)
            done = tr(lg, "ui.done", n=n_ok, total=len(seeds), path=out_dir)
            if proposal is not None:
                done += "  \n" + tr(lg, "ui.oom.proposal")
                upd = (
                    gr.update(value=proposal.pipeline_type),
                    gr.update(value=proposal.texture_size),
                    gr.update(value=proposal.decimation_target),
                )
            else:
                upd = (gr.skip(), gr.skip(), gr.skip())
            yield (
                done + "  \n" + vram_line(lg),
                *views_out,
                sheets,
                downloads,
                "\n".join(lines),
                *upd,
            )

        run.click(
            on_run,
            [
                image,
                remove_bg,
                preset,
                seed,
                n_seeds,
                resolution,
                texture,
                tex_guidance,
                faces,
                steps,
                lang,
            ],
            [status, *views, previews, files, report, resolution, texture, faces],
            concurrency_limit=1,
        )
    return demo


def launch(
    share: bool = False, port: int | None = None, lang: str | None = None, browser: bool = True
) -> None:
    runner = Runner()
    demo = build(runner, lang or i18n.get_lang())
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=port,
        share=share,
        inbrowser=browser,
        allowed_paths=[str(get_paths().outputs)],
    )
