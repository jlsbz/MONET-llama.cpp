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
