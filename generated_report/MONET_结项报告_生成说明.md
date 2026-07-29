# MONET 结项报告生成说明

## 1. 输出范围

本次工作仅在首次创建的 `generated_report/` 目录中生成以下四份 Markdown 文件：

1. `MONET_结项报告_初稿.md`
2. `MONET_结项报告_材料映射表.md`
3. `MONET_结项报告_待补充信息清单.md`
4. `MONET_结项报告_生成说明.md`

用于文档/PDF页面核对的临时 `_qa/` 子目录将在最终交付前删除，不作为交付物。

## 2. 采用的模板

采用模板：

- [`report/【上交-手机部-端侧大模型低比特量化】结项报告.md`](<../report/【上交-手机部-端侧大模型低比特量化】结项报告.md>)

选择理由：

- `report/` 目录中只有这一份候选模板；
- 文件名明确包含“结项报告”；
- 内容包含项目概述、项目验收及结论、总结与反思、过程/工作回顾、改进措施和项目组织过程资产等结项栏目；
- 报告初稿严格保留其标题、章节顺序、编号层级、字段名称和表格栏目，并在原有栏目内扩展内容。

模板本身是本次工作开始前已经存在的未跟踪文件。本次仅只读使用，没有修改。

## 3. 使用的源材料

### 3.1 项目公开说明与部署流程

- [根目录 README](../README.md)  
  用途：确认项目定位、Windows/WSL/Android 环境、HF→GGUF→量化→NDK→ADB/Termux→benchmark 的公开工作流，以及 Q4_K_M/Q5_K_M/Q6_K 等基础量化路径。

### 3.2 核心技术文档

- [中文项目说明手册](../docs/中文项目说明手册.md)  
  用途：梳理项目目标、五阶段工程链路、PyTorch 拟合、Monarch-dense/extra GGUF、loader/graph/GGML 修改路线、Android 部署和当时的完成/规划状态。

- [方案思路阐述](../docs/方案思路阐述.md)  
  用途：梳理从 CESD/GLAS/NGES、单聚合、layer-wise/block-wise、SVD，到 Monarch、FlashMonarch/流式执行、块稀疏和端侧测试的技术演进，并提取阶段性数据。

### 3.3 论文

- [投稿论文](../docs/投稿论文.pdf)  
  题目：*Breaking the Mobile Bandwidth Wall for LLMs via Semantic Block Decomposition*。  
  用途：提取 MONET 的正式问题定义、三项核心技术、实验设置、准确率、吞吐、内存流量、硬件利用率、消融和扩展性结果。

- [论文投稿截图](../docs/论文已投稿截图.png)  
  用途：确认论文在截图时的状态为 ATC '26 “Submitted”，稿件编号 #786，提交时间为 2026-06-11；没有据此推断录用或发表。

### 3.4 专利技术交底书

- [一种基于块分解的面向移动端大模型推理优化方法](../docs/技术交底书-一种基于块分解的面向移动端大模型推理优化方法.docx)  
  用途：提取移动端带宽问题、硬件感知块分解、带宽感知流式执行、成本感知块裁剪、应用场景和技术效果。

- [基于跨层块级子空间聚合的大型语言模型权重分解与低位表示方法](../docs/技术交底书-基于跨层块级子空间聚合的大型语言模型权重分解与低位表示方法.docx)  
  用途：提取跨层块级子空间分解（CL-BD）、全局损失对齐显著性（GLAS）、共享主干、层特异残差和差异化低位表示路线。

两份文件都按“技术交底书”处理。由于没有申请号、受理通知或授权信息，报告没有把它们写成已申请或已授权专利。

### 3.5 代码

- [PyTorch Monarch 拟合脚本](../scripts/fit_all_square_monarch_wikitext.py)
- [Monarch GGUF 转换脚本](../llama.cpp-monarch/convert_hf_to_gguf_monarch.py)
- [`llama-model.h`](../llama.cpp-monarch/src/llama-model.h)
- [`models/llama.cpp`](../llama.cpp-monarch/src/models/llama.cpp)
- [`llama-model.cpp`](../llama.cpp-monarch/src/llama-model.cpp)
- `llama.cpp-monarch/ggml/`、`src/` 和测试目录的精确关键词检索结果

用途：核验实际可见实现、默认参数、tensor 命名、模型结构和 loader 范围，并识别文档所述完整 runtime 与当前代码之间的边界。

### 3.6 测试与 Demo 图片

- [命令行基线](../docs/image-7.png)
- [命令行 Monarch](../docs/image-8.png)
- [Android App Monarch](../docs/image-9.png)
- [Android App 基线](../docs/image-10.png)
- `docs/image.png`—`docs/image-6.png`：模型结构、早期算法框架、相似度和 Monarch 结构示意

用途：核对可以直接从截图读取的模型大小、pp/tg、Prefill/Decode 数据，以及方案演进图示。

### 3.7 仓库与版本信息

- Git 远程地址：<https://github.com/jlsbz/MONET-llama.cpp>
- 仓库历史中可见的材料提交时间范围为 2026-05-21 至 2026-07-09，但这不是正式项目周期，报告未将其写成项目起止时间。
- `llama.cpp-monarch` 在仓库中作为整体加入，仓库没有可直接对照的上游原版快照，因此没有声称已经完整识别所有 MONET 相对原版的改动。

## 4. 重要归纳

### 4.1 项目主线

将多份材料统一归纳为以下主线：

```text
移动端带宽瓶颈识别
→ 早期跨专家/跨层子空间与极低比特探索
→ 单聚合到 layer-wise/block-wise
→ SVD 移动端适配问题
→ Monarch/语义块分解
→ 架构感知块大小
→ 带宽感知流式执行
→ 成本感知块裁剪
→ GGUF/llama.cpp/Android 部署与验证
```

### 4.2 MONET 与 `llama.cpp` 的关系

报告采用了分层表述：

- `llama.cpp` 提供 GGUF、量化、跨平台构建、命令行推理和 benchmark 基础；
- MONET 的仓库代码在其上增加 PyTorch 拟合、Monarch extra tensor 转换、模型结构和 loader 认领；
- 文档和论文进一步描述 graph、流式数据流、块裁剪与后端优化；
- 当前仓库不足以确认这些后续 runtime 能力全部已落到可见源码中。

### 4.3 成果状态

- 论文：只写“已投稿”，不写“已录用/发表”；
- 专利：只写“形成两份技术交底书”，不写“已申请/授权”；
- 应用：只确认端侧命令行和 App Demo，不写业务上线或用户规模；
- 验收：只写“具备技术审核基础”，不判定全部达标或正式结项。

### 4.4 测试结果口径

报告把三类结果分开：

1. 命令行截图：pp128/tg128 和模型显示大小；
2. App 截图：Prefill/Decode 单次读数；
3. 投稿论文：两台 Android 设备、GPU 后端的完整实验报告结果。

没有把不同后端、模型和设备的数字拼接成同一加速结论。

## 5. 主要冲突和不确定性

### 5.1 完整 runtime 实现状态

- 方案文档和论文描述了 Monarch/语义块分解的完整算子、流式执行和块选择；
- 中文手册第 11—13 节仍把 graph、自定义 GGML op、CPU/ARM kernel 和 Monarch-only GGUF 写成需要实现或规划内容；
- 当前代码检索未发现 `GGML_OP_MONARCH_LINEAR`、Monarch forward graph 调用或专用 CPU/ARM/GPU kernel。

处理方式：以当前代码可核验范围为准，把完整 runtime 标为待负责人确认。

### 5.2 命令行速度表述

- 方案文档使用“约 8 t/s 提升到约 10 t/s”的概括；
- 截图精确值为 pp128 7.87→9.83 t/s，tg128 5.63→6.24 t/s。

处理方式：正文采用截图精确值，并区分 pp 与 tg。

### 5.3 内存流量

- 方案文档列出 MONET/Monarch 总内存流量 186.42 MB/token；
- 投稿论文列出 118.8 MB/token；
- Dense 和 SVD 的 210.84/258.81 MB/token 在两处一致。

处理方式：论文栏引用投稿稿件中的 118.8 MB/token，同时把差异列为必须核对项。

### 5.4 精度表述

论文使用 “strict parity”“lossless”等概括，但表 2 中 MONET 相对 Dense 的 PPL/MMLU 有有限变化。

处理方式：报告列出代表性具体数字，使用“有限但非零的质量下降”，避免绝对化表述。

### 5.5 早期专利路线与当前 MONET 路线

跨层块级子空间聚合交底书以跨层 SVD 共享主干和 GLAS 低位残差为核心；移动端交底书和投稿论文更强调硬件感知块结构、流式执行与块裁剪。

处理方式：把前者写成前期算法探索/专利储备，把后者写成当前移动端系统路线，不合并为一个已经全部实现的单体方案。

## 6. 文档读取与质量检查说明

- Markdown、README 和源代码采用逐段/精确关键词方式读取；
- 两份 DOCX 通过文档结构读取全部非空段落、标题和表格；
- 当前环境缺少 LibreOffice 可执行文件，因此两份 DOCX 未完成页面渲染视觉检查；本次仅提取内容，不修改或交付 DOCX；
- 投稿论文共 12 页，已完成全文提取，并使用 PDF 渲染库生成页面图像，核对全部页面及关键表格/图表；
- 投稿截图和 4 张关键测试/Demo 图已进行视觉核对；
- 报告生成后将检查模板章节覆盖、相对链接、待补充标记、绝对化状态词和 Git 工作区变化。

## 7. 文件安全声明

开始工作前，Git 工作区已有以下 3 个未跟踪文件：

- `docs/技术交底书-一种基于块分解的面向移动端大模型推理优化方法.docx`
- `docs/技术交底书-基于跨层块级子空间聚合的大型语言模型权重分解与低位表示方法.docx`
- `report/【上交-手机部-端侧大模型低比特量化】结项报告.md`

本次将它们视为既有材料，没有恢复、覆盖、重命名、移动或删除。除 `generated_report/` 外，没有对 README、`docs/`、`report/`、`llama.cpp-monarch/`、`scripts/` 或其他原始目录写入内容。

最终 Git 状态复核已确认：上述 3 个既有未跟踪文件仍然存在，大小和修改时间与开始检查时一致；已跟踪文件和暂存区均无差异；本次只新增 `generated_report/` 下的 4 份 Markdown 交付物。
