# 문제 해결 / Troubleshooting

먼저 **PC 점검**을 해 보세요. 문제마다 무엇을 하면 되는지 알려 줍니다.
Run the **PC check** first; it says what to do for each problem.

```
trellis2 doctor          # Windows: 이 폴더에서 명령 프롬프트(cmd) / in this folder, Command Prompt
.\trellis2 doctor        # Windows PowerShell (Windows 11 기본 터미널 / default terminal)
./trellis2.sh doctor     # Ubuntu
```

웹 화면 맨 위의 **준비 상태**에도 같은 점검 결과가 나옵니다.
The **Readiness** panel at the top of the web page shows the same checks.

문제를 알릴 때는 `logs` 폴더의 가장 최근 파일을 [이슈](https://github.com/joonhyungbae/trellis2-lowvram/issues)에
첨부해 주세요. 토큰은 로그에 남지 않습니다.
When reporting a problem, attach the newest file in the `logs` folder to an issue.

---

## 설치 / Install

**설치가 몇 단계에서 멈췄다.** 실패한 단계 아래에 확인할 것이 나옵니다. 고친 뒤 `install.bat`(Ubuntu: `./install.sh`)을
다시 실행하면 끝난 단계와 받은 파일은 건너뜁니다.
*The failed step prints what to check. Run the installer again after fixing it; finished steps and
downloaded files are skipped.*

| 단계 / Step | 흔한 원인 / Common causes |
|---|---|
| 1 uv | 인터넷, 학교·회사 네트워크가 astral.sh·github.com 을 막음, 백신 / network blocks, antivirus |
| 2 PC 점검 / PC check | `[FAIL]` 줄을 먼저 해결. 드라이버를 새로 깔았으면 PC 재시작 / fix the `[FAIL]` lines; restart after a driver update |
| 3 TRELLIS.2 | github.com 접속 / access to github.com |
| 4 Python 패키지 / packages | 몇 GB 를 받음. 다시 실행하면 이어 받음. 디스크 30GB, 백신 예외, 경로에 한글·공백 없음 / several GB; re-run resumes; disk, antivirus, path |
| 5 3D 구성 요소 / components | github.com 접속, 백신이 .whl 을 막음 / github.com, antivirus blocking .whl |
| 6 모델 / models | DINOv3 승인 대기(아래), huggingface.co 접속, 디스크 / approval pending (below), network, disk |

**"실행 중인 창을 닫으라" 고 나온다.** 웹 화면(run.bat)이나 make3d 창이 켜져 있으면 파일이 잠겨 설치·업데이트가
중간에 실패합니다. 검은 창을 모두 닫고 다시 실행하세요.
*Close the web interface / make3d windows; Windows locks their files during install and update.*

**폴더 경로 경고.** `C:\trellis2-lowvram` 처럼 짧은 영문 경로에 두세요. 한글, 공백, 긴 경로에서는 일부 구성 요소가
실패할 수 있습니다. *Use a short English path without spaces.*

**run.bat 을 눌렀더니 "아직 설치되지 않았습니다".** `install.bat` 을 먼저 실행하세요.
*Run install.bat first.*

**업데이트.** `update.bat`(Ubuntu: `./update.sh`). ZIP 으로 받았다면 git 이 없으므로 새 ZIP 을 같은 폴더에 덮어쓴 뒤
`update.bat` 을 실행합니다. *Downloaded as ZIP: extract the new ZIP over this folder, then run update.bat.*

## Hugging Face, DINOv3

이미지 인코더 DINOv3 는 Meta 가 **사람이 직접 승인**합니다. 몇 시간에서 며칠 걸릴 수 있고, 그동안 다른 설치는 끝나
있습니다. 승인 메일을 받으면 웹 화면의 **모델 받기**를 누르세요.
*DINOv3 access is approved by hand (hours to days). Everything else is installed meanwhile; press
**Download models** in the web page after the approval email.*

기다리기 싫다면, 승인받은 사람(선생님)이 `trellis2 share-models` 로 만든 `shared-models` 폴더를 받아
`trellis2 import-models <폴더>` 를 실행하세요. 모델이 다 있으면 토큰 없이 실행됩니다(README 의 "모델 나눠 주기").
*To skip the wait, get a `shared-models` folder from someone approved and run
`trellis2 import-models <folder>`; with all models present no token is needed (README, "Sharing the models").*

| 메시지 / Message | 뜻과 할 일 / Meaning, what to do |
|---|---|
| 토큰이 없습니다 / No token | [토큰 만들기](https://huggingface.co/settings/tokens) → **Read** 토큰을 웹 화면이나 `trellis2 login` 에 입력 / create a Read token and enter it |
| 토큰이 올바르지 않습니다 / not valid | 잘못 복사했거나 지운 토큰. 새로 만들어 입력 / copied wrongly or deleted |
| 아직 승인되지 않았습니다 / not approved | [모델 페이지](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)에서 신청했는지 확인하고 기다리기 / request access and wait |
| 이 토큰은 DINOv3 를 읽을 수 없습니다 / cannot read DINOv3 | Fine-grained 토큰. 맨 위 **Read** 탭으로 토큰을 새로 만들기 / make a classic **Read** token |

## 웹 화면 / Web interface

**3D 화면이 회색으로 비어 있다.** 돌려 보는 3D 화면은 WebGPU 가 필요합니다(최신 Chrome, Edge). 그 아래의
**미리보기 이미지**와 **내려받기**는 어느 브라우저에서나 됩니다. 받은 GLB 는 Blender 같은 3D 프로그램에서 열 수 있습니다.
*The rotatable view needs WebGPU (recent Chrome or Edge). Previews and downloads work everywhere.*

**바로가기를 두 번 눌렀다.** 이미 열려 있으면 새로 띄우지 않고 열린 페이지를 브라우저로 엽니다. 모델을 메모리에 두 번
올리면 RAM 32GB PC 도 몹시 느려지기 때문입니다.
*A second launch opens the page that is already running instead of loading the models again.*

**멈추기를 눌렀는데 바로 안 멈춘다.** 지금 하는 단계가 끝나야 멈춥니다(길면 1분쯤). 멈추기 전에 끝난 모델은 남습니다.
*Stop takes effect when the current step ends.*

**1024 해상도를 골랐더니 모델을 받으라고 한다.** 1024 는 모델 파일이 약 5GB 더 필요합니다. 준비 상태의 **모델 받기**를 누르세요.
*1024 needs about 5GB more model files; press Download models in the Readiness panel.*

## 만들기 / Generating

**처음 한 번이 아주 느리다.** PC 에서 처음 실행할 때 계산 코드를 준비하느라 몇 분 더 걸립니다. 두 번째부터 빨라집니다.
*The first run on a PC compiles kernels and takes a few extra minutes.*

**늘 아주 느리다.** 다음을 확인하세요. *Check:*
- 웹 화면과 make3d(또는 웹 화면 두 개)를 동시에 켜 두지 않았는지. 둘 다 모델을 RAM 에 올립니다(하나에 약 20GB). /
  Two windows each hold the models in RAM (about 20GB each).
- 다른 큰 프로그램(게임, 영상 편집, 브라우저 탭 여러 개)을 닫았는지. / Close other heavy programs.
- Windows 는 그래픽 메모리가 차면 오류 대신 시스템 메모리를 빌려 쓰며 크게 느려집니다. 품질을 **빠르게**로, 텍스처를 1024 로
  낮춰 보세요. / On Windows a full GPU spills into system memory (slow, not an error); lower quality.

**그래픽 메모리가 모자랍니다(OOM).** 프로그램은 멈추지 않습니다. 웹 화면은 고급 설정을 한 단계 낮춰 두니 **3D 만들기**를
다시 누르세요. 명령어는 낮춘 설정으로 다시 할지 묻습니다(`-y` 면 자동).
*Out of graphics memory: the program keeps running and offers lower settings; press Make 3D again.*

**결과는 어디에?** `outputs/<날짜-시각>/` 폴더에 실행마다 따로 저장됩니다(GLB, `_obj` 폴더, 단계별 기록).
*Each run saves to its own `outputs/<date-time>/` folder.*

## 사진 / Pictures

- **아이폰 사진(HEIC)** 도 그대로 넣으면 됩니다. *iPhone HEIC photos work as is.*
- **세로로 찍은 사진**은 폰이 보여 준 방향대로 씁니다. *Portrait photos are used upright.*
- **그림으로 읽지 못했습니다**: 파일이 손상되었거나 그림이 아닙니다. 다른 프로그램에서 열린다면 JPG 나 PNG 로 다시 저장하세요.
  *Cannot read as a picture: re-save as JPG or PNG.*
- 배경이 단순하고 물체 하나가 화면을 채운 사진이 가장 잘 됩니다. 투명 배경 PNG 는 배경 지우기를 건너뜁니다.
  [좋은 입력 이미지 고르는 법](examples/README.md#tips-for-input-images--좋은-입력-이미지).
  *One object filling the frame on a plain background works best; transparent PNGs skip background removal.*
- 유리 용기는 겉면만 만들어지고 안에 비쳐 보이던 것은 사라집니다. *Glass containers: only the shell is made.*
