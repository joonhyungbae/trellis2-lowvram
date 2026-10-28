from pathlib import Path

from trellis2_lowvram.env_check import (
    GpuInfo,
    Status,
    check_disk,
    check_gpu,
    check_install_path,
    check_os,
    check_ram,
    worst,
)


def gpu(name="RTX 4070 Ti SUPER", vram=15.99, cc=(8, 9), driver_cuda=12080):
    return GpuInfo(name, vram, cc, "570.86", driver_cuda)


def by_id(checks):
    return {c.id: c for c in checks}


def test_no_gpu_fails():
    for value in (None, []):
        (c,) = check_gpu(value)
        assert c.status == Status.FAIL and c.msg == "check.gpu.missing"


def test_16gb_ampere_passes():
    c = by_id(check_gpu([gpu()]))
    assert c["gpu"].status == c["vram"].status == c["driver"].status == Status.OK


def test_rtx20_is_unsupported():
    c = by_id(check_gpu([gpu("RTX 2080 Ti", 11.0, (7, 5))]))
    assert c["gpu"].status == Status.FAIL and c["gpu"].fix == "fix.gpu.too_old"


def test_12gb_is_experimental_warning():
    assert by_id(check_gpu([gpu(vram=11.99)]))["vram"].status == Status.WARN


def test_old_driver_fails_with_link():
    c = by_id(check_gpu([gpu(driver_cuda=12040)]))["driver"]
    assert (
        c.status == Status.FAIL
        and c.params["driver_cuda"] == "12.4"
        and "nvidia.com" in c.params["url"]
    )


def test_picks_largest_gpu():
    c = by_id(check_gpu([gpu("small", 8.0), gpu("big", 23.99)]))
    assert c["gpu"].params["name"] == "big"


def test_ram_thresholds():
    assert check_ram(31.2).status == Status.OK
    assert check_ram(15.6).status == Status.WARN
    assert check_ram(7.8).status == Status.FAIL


def test_disk_thresholds():
    assert check_disk(100, Path("/x")).status == Status.OK
    assert check_disk(25, Path("/x")).status == Status.WARN
    assert check_disk(5, Path("/x")).status == Status.FAIL


def test_paths():
    assert check_install_path(Path("C:/trellis2-lowvram"), "Windows")[0].status == Status.OK
    ids = {c.id for c in check_install_path(Path("C:/Users/홍길동/My Folder"), "Windows")}
    assert {"path_ascii", "path_space"} <= ids
    assert check_install_path(Path("C:/" + "a" * 100), "Windows")[0].id == "path_long"
    assert check_install_path(Path("/" + "a" * 100), "Linux")[0].status == Status.OK


def test_os():
    assert check_os("Windows").status == Status.OK
    assert check_os("Darwin").status == Status.FAIL


def test_worst():
    ok, warn = check_ram(32), check_ram(16)
    assert worst([ok]) == Status.OK
    assert worst([ok, warn]) == Status.WARN
    assert worst([ok, warn, check_ram(4)]) == Status.FAIL


def test_windows_11_is_not_reported_as_10(monkeypatch):
    import platform

    from trellis2_lowvram.env_check import os_name

    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "release", lambda: "10")
    monkeypatch.setattr(platform, "version", lambda: "10.0.26100")
    assert os_name().startswith("Windows 11")
    monkeypatch.setattr(platform, "version", lambda: "10.0.19045")
    assert os_name().startswith("Windows 10")
