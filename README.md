# MONET:On-device LLaMA Inference on Android

This repository demonstrates how to deploy and benchmark LLaMA models on Android devices using changed `llama.cpp`.

The workflow supports:

* Windows + WSL2 development
* GGUF model conversion
* Quantization (Q4_K_M / Q5_K_M / Q6_K)
* Android NDK cross-compilation
* ADB deployment
* Termux-based execution
* Mobile benchmarking (`llama-bench`)
* PC vs Mobile inference demos

---

## Environment

### Host

* Windows 10/11
* WSL2 Ubuntu
* Android Studio
* Android NDK
* CMake
* Ninja
* ADB

### Device

* Android ARM64 phone
* Android 10+
* Termux (optional)

---

## Workflow Overview

```text
HuggingFace Model
        ↓
convert_hf_to_gguf.py
        ↓
GGUF (F16)
        ↓
llama-quantize
        ↓
Q4_K_M / Q5_K_M
        ↓
Android NDK Build
        ↓
ADB Push / Termux
        ↓
On-device Inference
```

---

## Build llama.cpp (WSL)

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

---

## Convert HF Model to GGUF

```bash
python3 convert_hf_to_gguf.py \
  /path/to/llama2-7b-hf \
  --outtype f16 \
  --outfile llama2-7b-f16.gguf
```

---

## Quantize Model

```bash
./build/bin/llama-quantize \
  llama2-7b-f16.gguf \
  llama2-7b-q4_k_m.gguf \
  Q4_K_M
```

---

## Android Cross Compilation

```powershell
cmake -S . -B build-android -G "Ninja" `
  -DCMAKE_BUILD_TYPE=Release `
  -DBUILD_SHARED_LIBS=OFF `
  -DGGML_OPENMP=OFF `
  -DCMAKE_TOOLCHAIN_FILE="$env:ANDROID_NDK\build\cmake\android.toolchain.cmake" `
  -DANDROID_ABI=arm64-v8a `
  -DANDROID_PLATFORM=android-28

cmake --build build-android -j4
```

---

## Deploy to Android

```powershell
adb push llama-cli /data/local/tmp/llama/
adb push llama2-7b-q4_k_m.gguf /data/local/tmp/llama/
```

Run:

```bash
./llama-cli -m llama2-7b-q4_k_m.gguf -p "Hello" -n 64
```

---

## Benchmark

```bash
./llama-bench \
  -m llama2-7b-q4_k_m.gguf \
  -p 128 \
  -n 128 \
  -t 8
```

---

## Termux Support

Optional Termux workflow:

```bash
pkg install openssh
sshd
```

Remote access:

```bash
ssh -p 8022 <termux-user>@<phone-ip>
```
---

## Notes

* Q4_K_M is recommended for mobile deployment.
* Disable battery optimization for Termux.
* Use `termux-wake-lock` during benchmarking.
* Static builds are recommended to avoid missing shared libraries (`libomp.so`, `libmtmd.so`).

---
