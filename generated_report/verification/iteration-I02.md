# Iteration I02 验证记录

- 日期：2026-07-30
- 工作分支：`codex/monet-completion`
- 代码提交：`d013fe9`
- 验证环境：Windows；Python 3.12.13；NumPy 2.3.5
- 本轮范围：方阵 Monarch forward 的 NumPy 数值参考契约与非循环稠密对照；不包含 C++/GGML graph、真实 GGUF 或 Android 设备。

## 实际执行

### 1. MONET Python 回归

```powershell
python -m unittest discover -s llama.cpp-monarch/tests -p "test_monarch*.py" -v
```

结果：通过，18/18。I02 新增的 6 项测试覆盖：

1. 直接分块 forward 与独立稠密物化结果一致；
2. 单块场景等价于先右乘 R、再右乘 L；
3. 单位因子下明确 `perm` 为 gather 语义；
4. 保留任意 leading dimensions，并至少使用 float32 计算；
5. 错误输入宽度被拒绝；
6. 非浮点及 NaN/Inf 输入被拒绝。

其余 12 项为 I01 的参数包与层名校验回归，均继续通过。

### 2. Python 语法编译

```powershell
python -m compileall -q \
  llama.cpp-monarch/monarch_tensor_validation.py \
  llama.cpp-monarch/monarch_reference.py \
  llama.cpp-monarch/convert_hf_to_gguf_monarch.py \
  llama.cpp-monarch/tests/test_monarch_tensor_validation.py \
  llama.cpp-monarch/tests/test_monarch_reference.py
```

结果：通过，无输出。

### 3. Git 差异格式检查

```powershell
git diff --check -- \
  llama.cpp-monarch/monarch_tensor_validation.py \
  llama.cpp-monarch/monarch_reference.py \
  llama.cpp-monarch/tests/test_monarch_reference.py
```

结果：通过；仅出现 Git 对工作区 LF/CRLF 转换策略的提示，不属于空白错误。

## 未执行与原因

- 未执行 C++/GGML 构建：I02 没有修改 C++，当前新增能力是 Python reference；它尚未接入 `llama.cpp` graph。
- 未执行 PyTorch 对照：当前环境未安装 PyTorch。参考语义已逐行对齐现有拟合脚本，并使用独立稠密物化路径交叉验证，但仍不替代真实拟合参数复核。
- 未执行 `.pt → GGUF → reader → loader`：仓库没有可执行的真实 Monarch `.pt` 参数包、对应 HF 权重和生成后 GGUF。
- 未执行 Android 测试：当前没有目标设备、APK/构建产物和固定模型。

## 验收边界

I02 关闭的是“建立可独立复核的方阵 Monarch 数值参考契约”这一小批次。`MONET-GAP-004` 的完整验收仍要求 `llama.cpp` forward graph 明确消费 L/R/perm、输出与本参考在约定容差内一致，且 dense 回退不回归；该项尚未关闭。
