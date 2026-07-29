# MONET 实现差距与验收闭环台账

本台账用于持续跟踪“材料承诺—代码实现—验证证据—正式验收”之间的差距。证据分类遵循任务约定：

- A 类：README、手册、结项报告或现有接口明确承诺；
- B 类：论文、专利或方案提出，且代码已有明显实现基础；
- C 类：仅研究材料提出，代码无明确基础或工程边界不清；
- D 类：依赖真实设备、模型/数据、业务材料、人员评价或正式验收。

状态只使用：未分析、已确认缺口、实现中、已实现待验证、已通过本地验证、待真实设备验证、待外部材料、暂缓、已关闭。

## 当前基线

- 建账日期：2026-07-30
- 基线分支及提交：`main` @ `ed5e11e`
- 持续迭代分支：`codex/monet-completion`
- 初始报告：[结项报告初稿](MONET_结项报告_初稿.md)、[材料映射表](MONET_结项报告_材料映射表.md)、[待补充信息清单](MONET_结项报告_待补充信息清单.md)

## 闭环台账

| ID | 来源 | 预期能力 | 当前代码状态 | 差距 | 优先级 | 本轮方案 | 验收标准 | 状态 | 证据 | 提交 |
|---|---|---|---|---|---|---|---|---|---|---|
| MONET-GAP-001 | A：手册第 6 节、既有 `--monarch-dir` 接口 | 在写入 GGUF 前可靠识别非法或冲突的 Monarch 参数包 | 原实现直接解引用 `.pt` 内容，缺少字段、类型、shape、有限值、permutation 和重复映射检查，且无专用测试 | 坏参数可能在晚期以模糊错误失败，或无支持层时仍表现为转换流程已启用 | P1 | 增加纯 NumPy 校验模块；converter 使用安全加载并在 writer 注册前校验；增加失败路径回归测试 | 受支持的合法输入通过；错误字段、层名、shape/dtype、NaN/Inf、非法 permutation 被拒绝；12 个单元测试、Python 语法编译和差异格式检查通过 | 已关闭 | [`monarch_tensor_validation.py`](../llama.cpp-monarch/monarch_tensor_validation.py)、[`convert_hf_to_gguf_monarch.py`](../llama.cpp-monarch/convert_hf_to_gguf_monarch.py)、[`test_monarch_tensor_validation.py`](../llama.cpp-monarch/tests/test_monarch_tensor_validation.py)、[I01 验证记录](verification/iteration-I01.md) | [`c2769bc`](https://github.com/jlsbz/MONET-llama.cpp/commit/c2769bc) |
| MONET-GAP-002 | A：手册第 6.4—6.6 节、初始报告模型格式栏 | 使用真实拟合参数生成 Monarch-extra GGUF，并可由 reader 核对 | converter 与校验逻辑存在；仓库没有可执行的真实 `.pt` 参数包、HF 权重和生成后 GGUF | 无法完成 `.pt → GGUF → reader` 集成回归，也无法核对“384 个 extra tensors”陈述 | P1 | 本轮完成不依赖模型的输入校验；待提供最小合法参数包和可复现模型后继续 | 固定模型/参数哈希；转换命令退出码 0；reader 中 tensor 名称、数量、shape、dtype 与预期一致；保留日志 | 待外部材料 | [持续版待补充清单](MONET_结项报告_待补充信息清单_持续更新版.md)、[手册第 6.4 节](../docs/中文项目说明手册.md) | — |
| MONET-GAP-003 | A：手册第 10 节、初始报告 loader 栏 | 修改版 `llama.cpp` 可加载带 Monarch extra tensor 的真实 GGUF | 代码中存在结构字段、可选 tensor 创建和 extra tensor 宽松检查 | 缺少与当前提交对应的构建日志、GGUF 样例、loader 日志和模型哈希 | P1 | 待 GAP-002 具备样例后执行主机端构建与加载回归 | 记录编译器/CMake 版本和构建命令；`llama-cli` 能加载固定哈希 GGUF；日志确认预期 projection 被认领且无未预期 tensor | 待外部材料 | [`llama-model.h`](../llama.cpp-monarch/src/llama-model.h)、[`models/llama.cpp`](../llama.cpp-monarch/src/models/llama.cpp)、[`llama-model.cpp`](../llama.cpp-monarch/src/llama-model.cpp) | — |
| MONET-GAP-004 | A：项目目标与手册第 11 节；B：投稿论文第 4 节 | forward graph 实际使用 Monarch L/R/perm 完成 attention projection | 当前仓库未检索到 graph 中使用 Monarch tensor 的调用；现有字段只到 loader 层 | 主流程仍不能凭现有代码证明执行 Monarch projection，核心 runtime 链路未闭合 | P0 | 下一轮先形成最小 CPU reference graph 设计与 dense 对照测试；实现前需确认 tensor 布局和数值语义 | 同一固定小矩阵下，reference Monarch forward 与 PyTorch/NumPy 结果在约定容差内一致；graph 明确消费 L/R/perm；dense 回退不回归 | 已确认缺口 | [持续版报告“当前实现边界”](MONET_结项报告_持续更新版.md)、[中文手册第 11 节](../docs/中文项目说明手册.md) | — |
| MONET-GAP-005 | B：论文/移动端交底书；C：手册第 12 节规划 | 具备 GGML 自定义 op、CPU/ARM/GPU kernel、流式 permutation 融合和块级执行能力 | 当前仓库未检索到 `GGML_OP_MONARCH_LINEAR` 或 MONET 专用 backend kernel | 论文/方案所述带宽优化和端侧加速无法由当前代码复现 | P1 | 在 GAP-004 reference 路径正确后分后端设计；不在缺少数值基线时直接大规模实现 | 自定义 op 具有 shape/type 检查；CPU reference 正确；目标后端测试通过；与 dense 的精度、吞吐和内存流量均有原始日志 | 已确认缺口 | [投稿论文](../docs/投稿论文.pdf)、[移动端交底书](../docs/技术交底书-一种基于块分解的面向移动端大模型推理优化方法.docx)、[中文手册第 12 节](../docs/中文项目说明手册.md) | — |
| MONET-GAP-006 | C：手册第 13 节规划 | 生成并加载不保留被替换 dense weight 的 Monarch-only GGUF | converter 只追加 extra tensor；loader/graph 仍依赖 dense 路径 | 缺少格式语义、回退规则和 runtime 前置能力 | P1 | 在 GAP-004/005 未闭合前暂不实现，避免产出无法运行的模型格式 | 有明确格式规范；转换后被替换 dense tensor 缺失；loader/graph 可运行；异常/回退路径测试通过；模型大小和哈希可复核 | 暂缓 | [`convert_hf_to_gguf_monarch.py`](../llama.cpp-monarch/convert_hf_to_gguf_monarch.py)、[中文手册第 13 节](../docs/中文项目说明手册.md) | — |
| MONET-GAP-007 | A：持续版结项报告和测试要求 | 建立覆盖 converter、loader、数值对齐和端到端流程的自动化回归 | I01 已覆盖转换前参数校验；没有真实 converter、loader 或推理测试 | 自动化证据只覆盖链路起点 | P1 | 后续按“真实参数转换→reader→loader→数值对齐”逐级补充 | 每一级有固定输入、可重复命令、明确断言和保存日志；硬件测试与本地测试分开标记 | 已确认缺口 | [持续版待补充清单“自动化测试”](MONET_结项报告_待补充信息清单_持续更新版.md)、[I01 验证记录](verification/iteration-I01.md) | [`c2769bc`](https://github.com/jlsbz/MONET-llama.cpp/commit/c2769bc)（仅起点校验） |
| MONET-GAP-008 | D：README、手册第 14 节、论文实验 | 在目标 Android 设备完成构建、加载、正确性和 benchmark 回归 | 仅有历史截图与材料陈述；本轮没有目标设备 | 无法核验设备、SoC、系统、温控、后端、线程和重复次数 | P1 | 先固定主机端样例和命令；具备设备后按统一模板复测 | 记录设备/系统/构建/模型哈希、预热和重复次数；保存原始日志；分别报告正确性、pp/tg 或 Prefill/Decode、内存、功耗和温升 | 待真实设备验证 | [README](../README.md)、[中文手册第 14 节](../docs/中文项目说明手册.md)、[持续版报告测试栏](MONET_结项报告_持续更新版.md) | — |
| MONET-GAP-009 | D：原始结项模板 | 逐项证明任务书/合同指标达标并形成正式验收结论 | 仓库无任务书、合同指标表、验收报告或会议纪要 | 无法判断“全部达标、通过验收、正式结项” | P0 | 保留明确待补充标记，不以代码提交替代正式验收 | 每项指标包含目标值、实测值、条件、证据编号、验收人和结论；有正式验收记录 | 待外部材料 | [持续版报告“项目验收及结论”](MONET_结项报告_持续更新版.md)、[持续版待补充清单](MONET_结项报告_待补充信息清单_持续更新版.md) | — |
| MONET-GAP-010 | D：论文、投稿截图、两份技术交底书 | 准确登记论文和专利最新正式状态 | 当前只能确认论文已投稿、两份材料为技术交底书 | 缺少录用/发表证明及专利申请/受理/授权材料 | P1 | 继续使用严格状态词，待项目负责人提供正式证明 | 论文提供录用/出版链接或明确仍在审；专利提供立案、申请、受理、公开或授权文件及编号 | 待外部材料 | [投稿截图](../docs/论文已投稿截图.png)、[持续版报告成果栏](MONET_结项报告_持续更新版.md) | — |
| MONET-GAP-011 | A：版本可审计要求 | 明确 MONET 相对选定上游 `llama.cpp` 基线的完整差异 | 仓库将 `llama.cpp-monarch` 整体加入，未记录上游 commit 映射 | 无法仅凭当前 Git 历史判定全部改动归属 | P1 | 由工程负责人确认上游 commit/tag，再生成只读差异清单；不猜测归属 | 固定上游 SHA；可复现 diff；MONET 修改按模块分类并由负责人确认 | 已确认缺口 | [持续版材料映射表](MONET_结项报告_材料映射表_持续更新版.md)、仓库提交历史 | — |
| MONET-GAP-012 | D：论文第 5 节、方案第 8—9 节 | 复现论文与截图中的性能、质量和内存流量结果 | 仓库只有论文汇总数字与截图，没有统一原始日志/脚本/模型哈希 | 118.8 与 186.42 MB/token 等口径差异尚未解释，最高加速无法独立复核 | P1 | 保留不同证据口径；待实验负责人提供复现包后统一评测 | 固定设备、后端、模型、量化、输入、温控和统计方法；原始日志可重新计算 PPL/MMLU、吞吐与流量 | 待外部材料 | [投稿论文](../docs/投稿论文.pdf)、[方案思路阐述](../docs/方案思路阐述.md)、[持续版材料映射表“冲突与差异”](MONET_结项报告_材料映射表_持续更新版.md) | — |

## 本轮关闭说明

Iteration I01 仅关闭 `MONET-GAP-001`。关闭依据是该事项的验收标准明确限定为转换前输入校验及其本地回归测试；未将 `MONET-GAP-002` 至 `MONET-GAP-008` 所要求的真实模型、loader、runtime 或 Android 验证合并计入，也未降低这些事项的验收标准。
