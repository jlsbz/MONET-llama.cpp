# Iteration I01 本地验证记录

## 验证范围

- Monarch 参数对象字段与兼容字段提取；
- LLaMA attention `q_proj/k_proj/v_proj/o_proj` 到 GGUF base 名称映射；
- L/R rank、shape、block 方阵、浮点 dtype 和有限值检查；
- permutation rank、长度、整数 dtype、重复值和越界值检查；
- 相关 Python 文件语法编译；
- 本轮代码差异格式检查。

## 环境

- 日期：2026-07-30
- 操作系统：Windows
- Python：3.12.13（Codex 工作区自带运行时）
- NumPy：2.3.5
- PyTorch：当前验证环境未安装
- 模型/参数/设备：未提供 HF 权重、真实 Monarch `.pt`、生成后 GGUF 或 Android 设备

## 执行记录

### 1. 单元测试

```text
python -m unittest discover -s llama.cpp-monarch/tests -p "test_monarch_tensor_validation.py" -v
```

结果：通过。共运行 12 个测试，失败 0、错误 0。

覆盖的测试类：

- `TestMonarchLayerMapping`
- `TestMonarchObjectExtraction`
- `TestMonarchArrayValidation`

### 2. Python 语法编译

```text
python -m compileall -q \
  llama.cpp-monarch/monarch_tensor_validation.py \
  llama.cpp-monarch/convert_hf_to_gguf_monarch.py \
  llama.cpp-monarch/tests/test_monarch_tensor_validation.py
```

结果：通过，退出码 0，无错误输出。

### 3. 差异格式检查

```text
git diff --check -- \
  llama.cpp-monarch/convert_hf_to_gguf_monarch.py \
  llama.cpp-monarch/monarch_tensor_validation.py \
  llama.cpp-monarch/tests/test_monarch_tensor_validation.py
```

结果：通过，未发现尾随空白或补丁格式错误。Git 提示工作区行尾将在后续写入时按 Windows 配置转换为 CRLF，该提示不属于检查失败。

## 未执行项目

以下项目没有被写为“通过”：

- `torch.load` 读取真实 `.pt` 参数包；
- 完整 `convert_hf_to_gguf_monarch.py` 转换；
- GGUF reader 对 tensor 名称、数量、shape、dtype 的核对；
- 修改版 `llama.cpp` 的构建和真实模型加载；
- PyTorch、NumPy 与 GGML 的数值对齐；
- forward graph、自定义 op、CPU/ARM/GPU kernel；
- Android 真机正确性、性能、内存、功耗和温升测试。

## 结论

本记录足以支持 `MONET-GAP-001` 的“转换前参数包校验与本地回归测试”关闭，不足以支持完整模型转换、runtime 或端侧验收结论。
