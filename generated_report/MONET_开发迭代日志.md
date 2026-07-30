# MONET 开发迭代日志

## Iteration I01：Monarch 参数包校验与转换前回归测试

- 日期：2026-07-30
- 基线分支及提交：`main` @ `ed5e11e`
- 工作分支：`codex/monet-completion`
- 本轮目标：为现有 Monarch-extra GGUF 转换入口补充安全、可诊断的参数包校验和可复现回归测试；不改变 CLI、GGUF tensor 命名或 dense weight 保留策略。
- 涉及的闭环台账 ID：`MONET-GAP-001`（关闭）；`MONET-GAP-002`、`MONET-GAP-007`（更新边界，未关闭）。
- 代码修改：
  - 新增 `llama.cpp-monarch/monarch_tensor_validation.py`，负责字段规范化、HF attention 层名映射、L/R/perm 结构与数值校验；
  - 更新 `convert_hf_to_gguf_monarch.py`，使用 `weights_only=True` 加载，检查 torch tensor/dtype、重复 GGUF base 和“无受支持层”情形；
  - 保持 `--monarch-dir`、`--monarch-dtype` 与 `*.monarch_l/r/perm` 命名不变。
- 新增或更新测试：新增 `llama.cpp-monarch/tests/test_monarch_tensor_validation.py`，共 12 个标准库 `unittest` 用例。
- 实际执行的验证：
  - `python -m unittest discover -s llama.cpp-monarch/tests -p "test_monarch_tensor_validation.py" -v`
  - `python -m compileall -q llama.cpp-monarch/monarch_tensor_validation.py llama.cpp-monarch/convert_hf_to_gguf_monarch.py llama.cpp-monarch/tests/test_monarch_tensor_validation.py`
  - `git diff --check`（限定本轮代码文件，提交后对文档再次检查）
- 验证结果：12/12 单元测试通过；三份 Python 文件语法编译通过；代码差异格式检查通过。详细记录见 [`verification/iteration-I01.md`](verification/iteration-I01.md)。
- 更新的说明手册章节：`docs/中文项目说明手册.md` 第 6.4 节“Monarch 参数包输入校验”，原生成/检查小节顺延为 6.5/6.6。
- 更新的结项报告章节：
  - 持续版报告的证据口径、产出量化、算法拟合与模型转换、测试与验证、验收矩阵、工作回顾；
  - 持续版材料映射表的 GGUF 转换、测试证据和关键代码索引；
  - 持续版待补充清单的“自动化测试”项目。
- 已知限制：
  - 本轮 Python 环境没有 PyTorch，未直接导入或执行完整 converter；
  - 仓库没有真实 Monarch `.pt` 参数包、HF 权重或生成后 GGUF；
  - 未验证 `llama.cpp` loader、forward graph、自定义 op/kernel 或数值对齐；
  - 既有完整 runtime 缺口未因本轮校验代码而关闭。
- 待真实设备验证事项：Android 构建、模型加载、命令行/App 正确性、Prefill/Decode、内存、功耗和温升均未执行。
- 代码提交：[`c2769bc`](https://github.com/jlsbz/MONET-llama.cpp/commit/c2769bc)
- 文档提交：[`49b03df`](https://github.com/jlsbz/MONET-llama.cpp/commit/49b03df)
- GitHub 同步：`codex/monet-completion` 已推送至 `origin`。
- Pull Request：未创建。GitHub App 创建动作返回 403；本机未安装 `gh`；应用内浏览器没有 GitHub 登录态。可由有权限的登录会话打开 [main...codex/monet-completion compare 页面](https://github.com/jlsbz/MONET-llama.cpp/compare/main...codex/monet-completion?expand=1) 创建 Draft PR。禁止使用强推或在文件中保存访问令牌。
- 下一轮建议：
  1. 提供最小真实 `.pt` 参数包和可复现 HF 模型，闭环 `.pt → GGUF → reader`；
  2. 在固定 GGUF 上补充主机端 loader 构建/加载回归和日志；
  3. 先设计并验证最小 CPU reference Monarch forward，再评估 graph、自定义 op 与后端 kernel。

## Iteration I02：Monarch forward 数值参考契约

- 日期：2026-07-30
- 基线分支及提交：`codex/monet-completion` @ `573edcb`
- 工作分支：`codex/monet-completion`
- 本轮目标：为 GAP-004 的后续 C++/GGML graph 接入建立可独立复核的方阵 Monarch 数值契约，先消除矩阵方向和 permutation gather/scatter 语义歧义；不在缺少数值基线和真实模型时直接实施大范围 runtime 改造。
- 涉及的闭环台账 ID：`MONET-GAP-013`（关闭）；`MONET-GAP-004`、`MONET-GAP-007`（更新为“实现中”，未关闭）。
- 代码修改：
  - 新增 `llama.cpp-monarch/monarch_reference.py`，实现 `R 块对角乘 → perm gather → L 块对角乘` 的直接 NumPy forward；
  - 在同一模块用显式块对角矩阵和 permutation matrix 独立物化等价稠密变换，作为非循环对照；
  - 从 `monarch_tensor_validation.py` 提取可供 layer 无关 reference 复用的 `validate_monarch_structure()`，保留原有 converter 校验行为。
- 新增或更新测试：新增 `llama.cpp-monarch/tests/test_monarch_reference.py`，共 6 个标准库 `unittest` 用例；与 I01 测试合并后共 18 项。
- 实际执行的验证：
  - `python -m unittest discover -s llama.cpp-monarch/tests -p "test_monarch*.py" -v`
  - `python -m compileall -q llama.cpp-monarch/monarch_tensor_validation.py llama.cpp-monarch/monarch_reference.py llama.cpp-monarch/convert_hf_to_gguf_monarch.py llama.cpp-monarch/tests/test_monarch_tensor_validation.py llama.cpp-monarch/tests/test_monarch_reference.py`
  - `git diff --check`（限定 I02 代码文件；文档同步后再次检查）
- 验证结果：18/18 单元测试通过，其中 I02 新增 6 项；五份 Python 文件语法编译通过；代码差异格式检查通过。详细记录见 [`verification/iteration-I02.md`](verification/iteration-I02.md)。
- 更新的说明手册章节：`docs/中文项目说明手册.md` 第 11.2 节“NumPy 数值参考契约（Iteration I02）”。
- 更新的结项报告章节：
  - 持续版报告的迭代状态、产出量化、当前实现边界、测试与验证、验收矩阵和工作回顾；
  - 持续版材料映射表的 forward 数值参考、关键代码索引和 runtime 差异说明；
  - 持续版待补充清单的“自动化测试”项目；
  - 闭环台账新增 `MONET-GAP-013`，并更新 `MONET-GAP-004`、`MONET-GAP-007`。
- 已知限制：
  - 本轮实现是 NumPy reference，不是 `llama.cpp` graph 或 GGML backend；
  - 当前环境没有 PyTorch，未使用真实拟合参数做 PyTorch/NumPy 三方对照；
  - 仓库没有真实 Monarch `.pt`、HF 权重或 GGUF，未执行 converter、reader、loader 和端到端推理；
  - `MONET-GAP-004` 仍要求 graph 明确消费 L/R/perm 并与本 reference 数值一致。
- 待真实设备验证事项：Android 构建、真实模型加载、命令行/App 正确性、Prefill/Decode、内存、功耗和温升均未执行。
- 代码提交：[`d013fe9`](https://github.com/jlsbz/MONET-llama.cpp/commit/d013fe9)
- 文档提交：[`2c80121`](https://github.com/jlsbz/MONET-llama.cpp/commit/2c80121)
- GitHub 同步：`codex/monet-completion` 已成功推送至 `origin`，远程提交为 `2c80121`。
- Pull Request：未创建。已确认远程无该分支的现有 open PR；GitHub App 创建 Draft PR 返回 403 `Resource not accessible by integration`。可由有权限的登录会话打开 [main...codex/monet-completion compare 页面](https://github.com/jlsbz/MONET-llama.cpp/compare/main...codex/monet-completion?expand=1) 创建 Draft PR；不得强推或在仓库中保存访问令牌。
- 下一轮建议：
  1. 以 `monarch_reference.py` 为 oracle，设计并实现只覆盖固定小矩阵的最小 C++/GGML graph 数值测试；
  2. 提供最小真实 `.pt` 参数包和可复现 HF 模型，闭环 `.pt → GGUF → reader`；
  3. 在固定 GGUF 上补充主机端 loader 构建/加载回归，再进入自定义 op/backend 优化。
