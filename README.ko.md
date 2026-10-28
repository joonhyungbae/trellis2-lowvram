# trellis2-lowvram

> **비공식 도구입니다. Microsoft 의 공식 프로젝트가 아니며** Microsoft 와 관계가 없습니다.
> [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2)(MIT)를 모델은 그대로 둔 채 감싸서 씁니다.

[English README](README.md)

사진 한 장을 색이 입혀진 3D 모델(GLB)로 바꿔 줍니다. 내 PC 에서 돌아갑니다.
같은 TRELLIS.2 4B 모델을 VRAM 16GB 급 NVIDIA 그래픽카드(RTX 30·40·50)에서 설치 한 번으로 쓰게 하는 것이 목표입니다.
Windows 와 Ubuntu 를 지원합니다.

![예시](docs/examples/typewriter.jpg)
![예시](docs/examples/mermaid.jpg)

> **현재 개발 중입니다(4단계 중 M3).** 설치, PC 점검, 명령어로 3D 만들기, 웹 화면까지 됩니다.
> 지금까지 Windows 11 의 RTX 3080 10GB 와, RTX 4090 에 14.5GB 상한을 건 모의 환경에서 쟀습니다([측정 결과](docs/BENCHMARKS.md)).
> **실제 16GB 그래픽카드에서는 아직 시험하지 않았습니다.** 더 많은 결과는 [예시](docs/examples/README.md)에 있습니다.

## 필요한 것

| | 조건 | 비고 |
|---|---|---|
| 운영체제 | Windows 10/11 또는 Ubuntu 22.04/24.04 | macOS 는 지원하지 않음 |
| 그래픽카드 | NVIDIA RTX 30·40·50 시리즈 | RTX 20 이하, AMD, Intel 은 지원하지 않음 |
| 그래픽 메모리 | 16GB | 12GB 이하는 실험적 |
| RAM | 32GB 권장 | 16GB 는 될 수도 있음 |
| 저장 공간 | 약 30GB (추정) | 프로그램과 모델 |
| NVIDIA 드라이버 | 570 이상 | 오래되었으면 설치 프로그램이 알려 줌 |

Python, CUDA, Visual Studio, conda 를 따로 설치할 필요는 **없습니다.**

## 설치

1. 이 저장소를 받습니다(초록색 **Code** 단추 → **Download ZIP**). `C:\trellis2-lowvram` 처럼 짧은 영문 경로에 풉니다.
2. **DINOv3 사용 신청을 먼저 해 둡니다.** 승인을 사람이 직접 해서 몇 시간에서 며칠이 걸립니다.
   [Hugging Face 계정](https://huggingface.co/join)을 만든 뒤
   [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) 페이지에서 신청 양식을 냅니다.
3. Windows 는 `install.bat` 을 더블클릭합니다. Ubuntu 는 `./install.sh` 를 실행합니다.
   설치 프로그램이 먼저 PC 를 점검하고, 필요한 것을 모두 이 폴더 안에 받습니다.
   마지막에 Hugging Face 토큰을 묻고([토큰 만들기](https://huggingface.co/settings/tokens), Read 권한), 바탕화면에 **trellis2-lowvram** 바로가기를 만듭니다.
   DINOv3 승인 전이면 모델 파일만 빼고 설치를 마칩니다. 승인 메일을 받은 뒤 웹 화면에서 **모델 받기**를 누르면 됩니다.
4. 실패하면 무엇을 확인할지와 로그 파일 경로가 나옵니다. [문제 해결](docs/TROUBLESHOOTING.md)을 보거나 로그를 이슈에 첨부해 주세요.

지우려면 폴더를 삭제하면 됩니다. 업데이트는 `update.bat` / `./update.sh` 입니다.

## PC 점검

```
trellis2 doctor        # Windows, 이 폴더에서 연 명령 프롬프트(cmd)
.\trellis2 doctor      # Windows PowerShell (Windows 11 기본 터미널)
./trellis2.sh doctor   # Ubuntu
```

`[FAIL]` 줄 아래에 무엇이 문제이고 무엇을 하면 되는지 나옵니다.

## 3D 모델 만들기

**웹 화면:** 바탕화면의 **trellis2-lowvram** 아이콘이나 `run.bat` 을 더블클릭합니다(Ubuntu 는 `./run.sh`). 브라우저에 이 PC 안에서만 열리는 페이지가 뜹니다.
사진(PNG, JPG, WEBP, 아이폰 HEIC)을 올리고 품질을 고른 뒤 **3D 만들기**를 누르면, 모델을 돌려 보고 GLB·OBJ 로 내려받을 수 있습니다.
seed 를 여러 개 한 번에 만들어 나란히 비교할 수 있고, **멈추기**는 지금 단계가 끝나면 멈추며 이미 끝난 모델은 남깁니다. 위쪽에서 한국어/English 를 바꿉니다.
이미 열려 있을 때 다시 실행하면 모델을 또 불러오지 않고 열린 페이지를 보여 줍니다.
페이지는 이 PC(127.0.0.1)에서만 열리며, 공개 링크는 `run.bat --share` 로 직접 요청할 때만 만듭니다.
돌려 보는 3D 화면은 WebGPU 가 되는 브라우저(최신 Chrome, Edge)가 필요합니다. 미리보기 이미지와 내려받기는 어느 브라우저에서나 됩니다.

![웹 화면](docs/images/web-ui-ko.png)

**끌어다 놓기 (Windows):** 사진 한 장 또는 여러 장을 `make3d.bat` 위에 끌어다 놓습니다. 사진마다 모델을 만들고(균형 품질), 끝나면 결과 폴더를 열어 줍니다. Ubuntu 는 `./make3d.sh photo.jpg`.

**명령어:**

```
trellis2 generate photo.jpg                      # Windows (이 폴더에서)
./trellis2.sh generate photo.jpg                 # Ubuntu
trellis2 generate photo.jpg --seeds 3            # 고를 수 있게 세 가지로
trellis2 generate photo.jpg --preset quality     # 텍스처 4096, 면 수 많게(더 느림)
```

결과는 실행마다 `outputs/<날짜-시각>/` 폴더에 따로 생깁니다. `.glb`(PBR 재질 포함)와 `_obj` 폴더(OBJ + 텍스처), 단계별 시간·그래픽 메모리 기록이 함께 나옵니다.
그래픽 메모리가 모자라면 어느 단계에서 멈췄는지 알려 주고 설정을 낮춰 다시 할지 묻습니다. PC 에서 처음 실행할 때는 몇 분 더 걸립니다.
[좋은 입력 이미지 고르는 법](docs/examples/README.md#tips-for-input-images--좋은-입력-이미지)도 참고해 주세요.

## 라이선스

이 저장소의 코드는 MIT 입니다. 모델 가중치는 포함하지 않고 설치할 때 공식 출처에서 받습니다.
일부 구성 요소는 따로 조건이 있습니다. 특히 **텍스처를 굽는 데 쓰는 nvdiffrast 는 비상업적 사용만 허용**합니다.
그래서 만든 모델은 공부, 연구, 비상업 전시에 씁니다. 전체 목록은 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) 에 있습니다.

## 문제가 생기면

[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) 에 설치 실패, DINOv3 승인, 3D 화면이 비어 있을 때, 느릴 때, 그래픽 메모리 부족을 정리했습니다.

## 개발자와 AI 에이전트용

[HANDOFF.md](HANDOFF.md) 부터 읽어 주세요. 설계 문서는 [docs/PLAN.md](docs/PLAN.md) 입니다.
