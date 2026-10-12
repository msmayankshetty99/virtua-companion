# Desktop distribution

The manual/tag-triggered `.github/workflows/release.yml` builds Windows x64 NSIS
and Linux x64 AppImage/deb packages with separate pinned llama.cpp CUDA/Vulkan
libraries, and a macOS arm64 (Apple silicon) DMG with a Metal library. Each
includes frozen Python and Electron; electron/native_backends.json lists which
bundles an OS ships. It uploads CI artifacts; it does not publish a GitHub release
automatically. Builds must pass loader/ABI smoke checks. These do not replace
GPU/model/audio tests on clean machines. `.github/workflows/native-smoke.yml`
builds the Metal bundle the same way (`build.py --smoke-bundle`) and generates
with a small model through it; no CI runner exercises the CUDA, Vulkan or HIP
bundles on a GPU.

ROCm (HIP) for Linux is opt-in: a manual run with the `hip` input adds a job that
builds the `hip` bundle in the digest-pinned `rocm/dev-ubuntu-22.04:7.2.1`
container (`.github/actions/hip-bundle`: `build.py --smoke-bundle hip`, then
`--check-bundle`, which loads it against ROCm without a GPU) and hands it to the
Linux leg as a tar (links kept), where `build.py` stages it from
`RIKO_OPTIONAL_BUNDLES` and release-checks it without ROCm; those packages are
named `-hip`. While it builds (hours: twelve GPU targets), the other legs wait.
Tag pushes and default runs never build it, so their size and time are unchanged.
rocm-compile.yml runs the same action when started by hand (Actions > ROCm compile); it is the
only proof, and it compiles and loads, never runs a model. The bundle needs ROCm 7
on the user's machine (not redistributed); setup offers it only beside an AMD GPU
that `rocminfo` reports when every GPU agent is one of its GPU_TARGETS and ROCm 7's
`libamdhip64.so.7` is installed; Vulkan stays the default for AMD otherwise.

macOS: requires macOS 14 or later. Without signing secrets the app is signed ad
hoc with the hardened runtime, so Gatekeeper blocks the first open of a download;
approve it under System Settings > Privacy & Security ("Open Anyway"). With the
repository secrets MAC_CSC_LINK (base64 Developer ID Application .p12) and
MAC_CSC_KEY_PASSWORD the app is signed with that identity, and with MAC_NOTARY_KEY
(App Store Connect API key .p8 text), MAC_NOTARY_KEY_ID and MAC_NOTARY_ISSUER it
is also notarized. Riko asks for microphone access at launch; the backend's voice
input is attributed to Riko.app.

Windows lets the user select the application installation directory. The private
data directory is selected separately by the first-launch wizard on every system
(AppImage has no installer). Its location is recorded under Electron userData.
Upgrades/uninstall must not remove it. Hugging Face cache, probe data, memories,
config and logs use this directory. Local GGUF selections remain at their original
paths. No model weights, personal config, .env, memories or unlicensed avatar/voice
assets are copied from a developer's machine.

CUDA is preferred when nvidia-smi detects NVIDIA; Vulkan is also included, and
ROCm (HIP) in a `-hip` Linux package when ROCm finds an AMD GPU. Users need
compatible GPU drivers (and ROCm 7 for HIP), not a build toolchain. No silent
backend fallback.
macOS ships only Metal, which needs no driver or toolchain.

The CUDA and Vulkan bundles build ggml's backends as modules (`GGML_BACKEND_DL`,
from electron/native_backends.json) with one CPU module per x86 level
(`GGML_CPU_ALL_VARIANTS`, as llama.cpp's own releases): riko-native loads them from
its own folder and ggml picks the best CPU build the processor runs, so CPUs
without AVX2/FMA/F16C/BMI2 no longer die with an illegal instruction. A GPU module
that cannot load or finds no device leaves llama.cpp on the CPU, and riko-native
says so in its notes. `--release-check` proves a CPU module loads from the bundle
and every module loads with its dependencies from there.

`build.py` refuses to run over an earlier `.native/llama.cpp-release`,
`release-stage/` or `release-build/`; it keeps each library once with its versioned
links, moves PyInstaller's output into `release-stage/backend`, and hashes each
shipped file once. electron-builder's extraResources copy recreates those links.
Users get `notices/NOTICE.md` (tools/release/NOTICE.md), not this file.

Python torch is CPU-only in these packages (Julia, embeddings and probe training);
CUDA main-model inference is provided by the native llama.cpp backend, not torch.
Linux packages currently require PortAudio, libsndfile and an audio/desktop stack;
the Vulkan module also needs the system Vulkan loader. CUDA runtime/cuBLAS are
copied from the toolkit; do not redistribute NVIDIA drivers. Review NVIDIA's
redistributable list/EULA, Python package licenses, Vulkan SDK/loader notices and
avatar/model/voice licenses before public distribution. Signing certificates and
publishing credentials are not configured for Windows; unsigned Windows installers
will trigger trust warnings.

GPT-SoVITS is an optional externally installed API executable. Its executable,
argument list, URL and reference audio are editable in Settings. Auto-start is
explicitly opt-in in packaged mode; the app terminates its owned direct process,
not unrelated services or process trees. Service readiness is not guaranteed at
launch. External MCP tools, FFmpeg media decoding and Discord must also be tested
and provisioned separately; packaging Python does not install arbitrary tools.

Release dependency versions (except pinned llama.cpp/build tools) are not yet fully
locked. CI builds need clean-machine acceptance tests and a license/SBOM audit
before being described as production-ready. CUDA/Vulkan hardware execution is not
verified by the source tests. No installer or GPU build was run on the developer
machine while adding this pipeline.
