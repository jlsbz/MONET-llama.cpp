# MONET 结项报告材料映射表

## 1. 映射原则

- 报告结构以 [`report/【上交-手机部-端侧大模型低比特量化】结项报告.md`](<../report/【上交-手机部-端侧大模型低比特量化】结项报告.md>) 为唯一主模板。
- “代码确认”仅指当前仓库中能直接定位的行为；因仓库未保留可比较的原版 `llama.cpp` 基线，不能据此完整枚举 MONET 相对上游的全部差异。
- 论文结果、手册结果、方案说明和截图结果分别保留其证据口径，不把不同设备、后端和模型下的数字合并成同一验收结果。
- 专利材料按“技术交底书”登记；论文按“已投稿”登记。
- 持续迭代中的“已通过本地验证”只覆盖对应自动化检查，不外推为真实 GGUF、完整推理运行时或 Android 设备验收结论。

## 2. 模板—材料映射表

| 模板章节或字段 | 已填写内容概述 | 主要依据文件 | 证据位置 | 完整程度 | 待确认事项 |
|---|---|---|---|---|---|
| 标题 | 沿用模板标题“【上交-手机部-端侧大模型低比特量化】结项报告” | [结项报告模板](<../report/【上交-手机部-端侧大模型低比特量化】结项报告.md>) | 首行 | 已充分填写 | 正式立项名称是否与模板标题完全一致 |
| 汇报人 | 使用具体待补充标记 | 结项报告模板 | 页首字段 | 缺少材料 | 正式汇报人姓名及角色 |
| 部门 | 使用具体待补充标记 | 结项报告模板 | 页首字段 | 缺少材料 | 一级/二级/三级部门 |
| 日期 | 使用具体待补充标记 | 结项报告模板 | 页首字段 | 缺少材料 | 正式汇报日期 |
| 一、项目概述—一句话最大产出 | 概括算法、模型转换、GGUF 扩展、loader、端侧验证和科研成果链路 | [README](../README.md)、[方案思路阐述](../docs/方案思路阐述.md)、[中文项目说明手册](../docs/中文项目说明手册.md)、关键代码 | 方案第 5—11 节；手册第 0—14 节；代码见下方索引 | 已充分填写 | 是否有完整 runtime 位于其他分支或私有仓库 |
| 项目产出量化记录 | 统计 7 个关键代码文件、2 个 MONET 专用测试文件、1 篇投稿论文、2 份技术交底书、2 份技术文档、12 张过程/投稿图片，并列出截图和论文中的主要结果 | [docs 目录](../docs/)、[scripts 目录](../scripts/)、[修改版 llama.cpp](../llama.cpp-monarch/) | 文件清单；论文第 5 节；`image-7.png`—`image-10.png`；I01/I02 验证记录 | 已部分填写 | 正式成果统计口径、是否还有未入库成果 |
| 1.项目产出—背景及问题 | 移动端 LLM 的 LPDDR/缓存/功耗限制；量化、稀疏、SVD 的移动端局限 | [方案思路阐述](../docs/方案思路阐述.md)、[投稿论文](../docs/投稿论文.pdf)、[移动端技术交底书](../docs/技术交底书-一种基于块分解的面向移动端大模型推理优化方法.docx) | 方案第 1、4 节；论文第 1、3 节；交底书第 1.1—1.2 节 | 已充分填写 | 无 |
| 1.项目产出—总体目标及应用场景 | 在 `llama.cpp` 支持 Monarch 结构化压缩并验证 Windows/WSL/Android；手机本地、离线、隐私场景 | [中文项目说明手册](../docs/中文项目说明手册.md)、[README](../README.md)、移动端技术交底书 | 手册第 0、1、14 节；README Environment/Workflow；交底书第 2.1 节 | 已充分填写 | 正式任务书中的目标表述和优先场景 |
| 1.项目产出—早期路线 | CESD、GLAS、NGES；单聚合到 layer-wise/block-wise；Qwen3-VL 早期实验 | [方案思路阐述](../docs/方案思路阐述.md)、[跨层技术交底书](../docs/技术交底书-基于跨层块级子空间聚合的大型语言模型权重分解与低位表示方法.docx) | 方案第 2—3 节；交底书第 2.3.1—2.3.3 节 | 已充分填写 | 早期实验原始记录及与最终验收范围的关系 |
| 1.项目产出—最终/当前技术路线 | 架构感知块分解、带宽感知流式执行、成本感知离线块裁剪 | [投稿论文](../docs/投稿论文.pdf)、[方案思路阐述](../docs/方案思路阐述.md)、移动端技术交底书 | 论文第 4 节及算法 1；方案第 5—7 节；交底书第 2.3 节 | 已充分填写 | 最终实现版本与论文描述是否一致 |
| 1.项目产出—PyTorch 拟合 | 方阵 attention projection；WikiText-2 激活拟合；保存 L/R/perm 和误差报告 | [拟合脚本](../scripts/fit_all_square_monarch_wikitext.py)、[中文项目说明手册](../docs/中文项目说明手册.md) | 脚本第 21—147、159—199、214—265、360—543、550—889 行；手册第 3—4 节 | 已充分填写 | 实际运行参数、输出目录、完整 `fit_report` |
| 1.项目产出—GGUF 转换 | `--monarch-dir`/`--monarch-dtype`；HF 层名映射；追加 L/R/perm extra tensor；写入前参数包校验和重复映射检查 | [转换脚本](../llama.cpp-monarch/convert_hf_to_gguf_monarch.py)、[校验模块](../llama.cpp-monarch/monarch_tensor_validation.py)、[中文项目说明手册](../docs/中文项目说明手册.md) | 转换脚本 `load_monarch_obj`/`load_all_monarch_tensors`；校验模块；手册第 6.4 节 | 已充分填写 | 成功生成 384 tensors 的 reader dump、GGUF 样例与哈希 |
| 1.项目产出—转换校验测试 | 12 个单元测试覆盖层名映射、字段兼容、shape/dtype/有限值与 permutation 失败路径 | [测试文件](../llama.cpp-monarch/tests/test_monarch_tensor_validation.py)、[验证记录](verification/iteration-I01.md) | `TestMonarchLayerMapping`、`TestMonarchObjectExtraction`、`TestMonarchArrayValidation`；提交 `c2769bc` | 已充分填写 | 真实 PyTorch `.pt` 参数包与完整 converter 集成测试 |
| 1.项目产出—forward 数值参考 | NumPy 直接分块 forward 与独立稠密物化共同固定 `R → gather(perm) → L` 的行向量语义；6 个单元测试覆盖一致性、维度、dtype 和失败路径 | [参考实现](../llama.cpp-monarch/monarch_reference.py)、[测试文件](../llama.cpp-monarch/tests/test_monarch_reference.py)、[I02 验证记录](verification/iteration-I02.md) | `monarch_linear_reference`、`materialize_monarch_dense_reference`、`TestMonarchReference`；提交 `d013fe9` | 已充分填写 | PyTorch 实际拟合参数对照；C++/GGML graph 数值对齐 |
| 1.项目产出—模型结构和 loader | `llama_monarch_weight`、Q/K/V/O 字段、可选 tensor 创建、额外 tensor 宽松检查 | [`llama-model.h`](../llama.cpp-monarch/src/llama-model.h)、[`models/llama.cpp`](../llama.cpp-monarch/src/models/llama.cpp)、[`llama-model.cpp`](../llama.cpp-monarch/src/llama-model.cpp) | `llama-model.h` 第 223—235、289—294 行；`models/llama.cpp` 第 57—110 行；`llama-model.cpp` 第 1453—1455 行 | 已充分填写 | loader 的实际编译/加载日志 |
| 1.项目产出—完整 graph/算子/kernel | I02 已形成 NumPy 数值基线；报告继续说明当前仓库未检索到 `GGML_OP_MONARCH_LINEAR`、forward graph 使用或 MONET 专用 CPU/ARM/GPU kernel | 当前 `llama.cpp-monarch/src` 与 `ggml` 源码；[NumPy 参考实现](../llama.cpp-monarch/monarch_reference.py)；[中文项目说明手册](../docs/中文项目说明手册.md) | 全仓库精确检索；手册第 11.2—12 节 | 已部分填写 | 完整实现是否在其他分支/仓库；若无，应按数值基线实现 graph 并确定后端计划 |
| 1.项目产出—命令行测试 | 基线/Monarch 模型大小、pp128、tg128 读数及相对变化 | [基线截图](../docs/image-7.png)、[Monarch 截图](../docs/image-8.png)、[方案思路阐述](../docs/方案思路阐述.md) | 图片表格；方案第 8.1 节 | 已部分填写 | 设备、OS、后端、commit、完整命令、模型哈希、预热与重复次数 |
| 1.项目产出—App 测试 | App Prefill/Decode 读数及相对变化；仅作为 Demo 证据 | [Monarch App 截图](../docs/image-9.png)、[基线 App 截图](../docs/image-10.png)、方案思路阐述 | 图片底部性能字段；方案第 8.2 节 | 已部分填写 | APK/源码、相同 Prompt、重复测试、统计口径 |
| 1.项目产出—论文实验 | 两台 Android 设备、GPU 后端、模型/数据集、PPL/MMLU、吞吐、内存流量、利用率、消融与扩展性 | [投稿论文](../docs/投稿论文.pdf) | 第 5 节，表 1—2、图 8—16 | 已部分填写 | 原始日志、脚本、硬件具体型号、模型文件、复现实验和第三方审核 |
| 1.项目产出—论文成果状态 | 表述为 ATC '26 已投稿，不表述为录用/发表 | [论文稿件](../docs/投稿论文.pdf)、[投稿截图](../docs/论文已投稿截图.png) | 稿件首页；截图顶部和 “Submitted” 状态 | 已充分填写 | 当前审稿/录用状态；最终版本及公开链接 |
| 1.项目产出—专利成果状态 | 两份技术交底书的主题和技术范围；不宣称申请或授权 | 两份 DOCX 技术交底书 | 两份交底书首页表格、背景技术和第 2.3 节 | 已充分填写 | 是否立案、申请、受理或授权；申请号/公开号/专利号 |
| 1.项目产出—是否达标/通过/结项 | 说明技术成果已形成，但缺少任务书和验收报告，不能判定正式通过 | 模板、全部技术材料、仓库缺口 | 报告综合判断 | 仅能提供框架 | 任务书指标、验收报告、验收结论和遗留项关闭标准 |
| 2.项目应用—业务部门应用 | 仅确认 Android/Termux/ADB 与 App 原型验证；未宣称业务上线 | README、方案第 8 节、截图 | README Deploy/Benchmark；方案第 8 节；`image-7`—`image-10` | 已部分填写 | 业务部门、产品版本、用户/试用数据、负责人 |
| 2.项目应用—战投产投应用 | 保留具体待补充项 | 无 | 无 | 缺少材料 | 是否用于尽调、投资评估或产业合作 |
| 2.项目应用—品牌应用 | 仅确认投稿和研发型 Demo；未宣称品牌应用 | 投稿截图、Demo 截图 | 图片 | 仅能提供框架 | 宣传活动、会议展示、对外发布与品牌证明 |
| 2.项目应用—其他 | 归纳为端侧结构化压缩和 `llama.cpp` 二次开发技术储备 | 全部技术材料 | 综合归纳 | 已部分填写 | 是否开源发布、跨团队复用或产品化 |
| 3.项目成本阐述 | 使用具体待补充标记 | 无 | 无 | 缺少材料 | 合同金额、设备/算力费用、已付款、人月投入和偏差 |
| 4.Demo或实物展示 | 链接 4 张命令行/App 截图、README 和中文手册 | `image-7`—`image-10`、README、中文项目说明手册 | 对应文件 | 已部分填写 | 可运行 Demo、APK、录屏、演示设备、脚本 |
| 5.高校老师评价 | 使用具体待补充标记 | 无 | 无 | 缺少材料 | 老师姓名职务、评价文字、日期和截图 |
| 二、项目验收及结论 | 按目标定义、算法、转换/加载、runtime、端侧验证、科研成果、管理成本建立验收矩阵 | 全部材料 | 报告综合矩阵 | 已部分填写 | 正式验收组意见、时间、结论、遗留问题 |
| 三、总结与反思 | 总结算法—系统协同价值，并指出实现与证据链不统一的主要风险 | 方案、论文、手册、代码和截图 | 综合归纳 | 已充分填写 | 项目负责人确认反思口径 |
| 过程回顾 | 归纳 SVD 不适配、单聚合精度、代码/文档缺口、测试口径不统一 | 方案第 2—6、9—10 节；手册第 3、11—13 节；代码检索 | 对应章节 | 已充分填写 | 负责人补充组织和协作层面的过程问题 |
| 工作回顾 | 列出问题定义、路线调整、成果沉淀方面的优点及证据/实现不足 | 全部材料 | 综合归纳 | 已充分填写 | 项目团队审核措辞 |
| 改进措施/经验 | 建议验收矩阵、状态清单、可复现 benchmark、原始日志、技术分支区分和严格成果状态词 | 全部材料 | 综合归纳 | 已充分填写 | 责任人和完成时间 |
| 项目组织过程资产—项目 charter | 链接待补充 | 无 | 无 | 缺少材料 | 立项书、任务书或项目章程 |
| 项目组织过程资产—评审记录 | 链接待补充 | 无 | 无 | 缺少材料 | 方案/阶段/测试/验收评审纪要 |
| 项目组织过程资产—Demo | 链接现有截图和文档 | README、方案、`image-7`—`image-10` | 对应文件 | 已部分填写 | 可执行包、视频、演示说明 |

## 3. 关键代码证据索引

| 模块 | 本地文件 | GitHub 链接 | 可确认内容 | 当前边界 |
|---|---|---|---|---|
| 激活感知拟合 | [`scripts/fit_all_square_monarch_wikitext.py`](../scripts/fit_all_square_monarch_wikitext.py) | [GitHub](https://github.com/jlsbz/MONET-llama.cpp/blob/main/scripts/fit_all_square_monarch_wikitext.py) | 方阵 `MonarchLinear`、WikiText-2 激活采集、逐层拟合、L/R/perm 和误差报告保存 | 默认仅 q/k/v/o 方阵；不处理矩形 MLP |
| GGUF 扩展 | [`convert_hf_to_gguf_monarch.py`](../llama.cpp-monarch/convert_hf_to_gguf_monarch.py) | [GitHub（本轮分支）](https://github.com/jlsbz/MONET-llama.cpp/blob/codex/monet-completion/llama.cpp-monarch/convert_hf_to_gguf_monarch.py) | 从 `.pt` 安全读取 L/R/perm，校验后追加 extra tensor，支持 f32/f16 | 仍保留 dense 权重；无 `--monarch-only`；真实转换待模型材料 |
| 参数包校验 | [`monarch_tensor_validation.py`](../llama.cpp-monarch/monarch_tensor_validation.py) | [GitHub（本轮分支）](https://github.com/jlsbz/MONET-llama.cpp/blob/codex/monet-completion/llama.cpp-monarch/monarch_tensor_validation.py) | 规范化字段、HF→GGUF 名称映射、L/R/perm 结构和数值约束 | 只覆盖当前方阵 attention 映射 |
| 参数包校验测试 | [`test_monarch_tensor_validation.py`](../llama.cpp-monarch/tests/test_monarch_tensor_validation.py) | [GitHub（本轮分支）](https://github.com/jlsbz/MONET-llama.cpp/blob/codex/monet-completion/llama.cpp-monarch/tests/test_monarch_tensor_validation.py) | 12 个单元测试本地通过；提交 `c2769bc` | 未导入 PyTorch/完整 converter，未使用真实 `.pt` |
| Monarch 数值参考 | [`monarch_reference.py`](../llama.cpp-monarch/monarch_reference.py) | [GitHub（本轮分支）](https://github.com/jlsbz/MONET-llama.cpp/blob/codex/monet-completion/llama.cpp-monarch/monarch_reference.py) | 直接分块 forward、独立稠密物化、公共结构校验；提交 `d013fe9` | 仅 NumPy 方阵 reference，不是 GGML runtime |
| Monarch 数值参考测试 | [`test_monarch_reference.py`](../llama.cpp-monarch/tests/test_monarch_reference.py) | [GitHub（本轮分支）](https://github.com/jlsbz/MONET-llama.cpp/blob/codex/monet-completion/llama.cpp-monarch/tests/test_monarch_reference.py) | I02 新增 6 项，完整 MONET Python 测试集 18/18 本地通过 | 未使用真实 `.pt`、PyTorch 或 C++ graph |
| 模型字段 | [`llama-model.h`](../llama.cpp-monarch/src/llama-model.h) | [GitHub](https://github.com/jlsbz/MONET-llama.cpp/blob/main/llama.cpp-monarch/src/llama-model.h) | `llama_monarch_weight` 及 Q/K/V/O 字段 | 仅数据结构 |
| Tensor 创建/认领 | [`models/llama.cpp`](../llama.cpp-monarch/src/models/llama.cpp) | [GitHub](https://github.com/jlsbz/MONET-llama.cpp/blob/main/llama.cpp-monarch/src/models/llama.cpp) | 块大小 64、可选 L/R/perm tensor、enabled 判定 | 未发现 forward graph 使用 |
| Extra tensor 兼容 | [`llama-model.cpp`](../llama.cpp-monarch/src/llama-model.cpp) | [GitHub](https://github.com/jlsbz/MONET-llama.cpp/blob/main/llama.cpp-monarch/src/llama-model.cpp) | `done_getting_tensors(true)` | 注释明确为临时宽松处理 |

## 4. 材料冲突与差异记录

| 主题 | 材料 A | 材料 B/代码 | 处理方式 |
|---|---|---|---|
| 完整 runtime 状态 | 方案第 10.1 节称已完成 `llama.cpp/GGML` 自定义算子支持；论文称已实现分解、流式执行和块选择 | 当前代码确认拟合、GGUF extra tensor、模型字段、loader 和 NumPy reference；未发现 `GGML_OP_MONARCH_LINEAR`、graph 调用或专用 kernel；中文手册第 11.2 节明确 reference 不等于 runtime | 报告采用代码可核验边界，把数值基线写为已本地验证，同时把完整 runtime 标为待确认，不直接认定已完成 |
| 命令行速度表述 | 方案称 token 速度约由 8 t/s 提升至 10 t/s | 截图精确值为 pp128 7.87→9.83，tg128 5.63→6.24 | 报告优先采用截图精确读数，并指出 pp/tg 口径 |
| App 加速 | 方案只做定性描述 | 截图为 Prefill 8.70→10.66、Decode 6.62→7.16，且两次生成内容不同 | 作为单次 Demo 证据，不作为正式统计结论 |
| 总内存流量 | 方案第 9.3 节列出 Dense/SVD/Monarch 为 210.84/258.81/186.42 MB/token | 投稿论文第 5.3 节写 MONET 为 118.8 MB/token，Dense/SVD 为 210.84/258.81 | 优先在论文成果栏引用正式投稿稿件结果，同时把差异列为必须核对项 |
| 加速幅度 | 本地命令行/App 截图显示约 1.08×—1.25× | 论文报告最高 2.42× Prefill、2.52× Decode | 不视为直接冲突；两者后端、设备、模型和优化版本可能不同，报告分开列示并要求补齐测试条件 |
| 精度表述 | 论文摘要/正文使用 “strict parity”“lossless”等表述 | 论文表 2 显示 MONET 相对 Dense 有有限 PPL/MMLU 变化 | 报告使用具体数字和“有限但非零的质量下降”，避免“完全无损” |
| 专利状态 | 仓库含两份技术交底书 | 无申请号、受理通知或授权材料，且交底书部分人员/案号字段仍为空 | 仅计为技术交底成果 |
| 论文状态 | 投稿截图显示 “Submitted” | 无录用通知或正式出版信息 | 仅写“已投稿”，不写“录用/发表” |
