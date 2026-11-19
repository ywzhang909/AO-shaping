# -*- coding: utf-8 -*-
"""AO-shaping 术语统一引擎（2026-10-16；2026-10-22 扩展 .py，issue #187）

依据 docs/terminology_glossary.md 的术语对照表，对仓库 markdown 文档与
Python 文件做专业化替换。
规则：先特殊（长/组合）后一般（短），围栏代码块与 mermaid 图跳过，行内
`code` 保护。

.py 扩展（2026-10-22，issue #187）：.py 的中文叙述（docstring / 字符串
字面量 / 注释）进入同一套中文术语规则；f-string 的 {expr} 部分保护
（表达式内可能是与禁词同名的标识符/字典键）。pupil / bake-off 的英文
夹注规则**不适用**于 .py（§2 纯英文 DO-NOT-TOUCH + 保护
`--pupil-diameter` 等 CLI 标识符）。.py 排除集：本文件自身（规则表用
禁词作源串，扫描会自我改写、失去幂等，同 .md 侧 EXCLUDE_NAMES）、
禁词守卫测试（banned 元组是测试数据，替换即反转守卫）、libs/（vendored
第三方库）、call_graph_wiki（生成文档）。

用法：
  python3 term_unify_20261016.py          # dry-run（默认，恒退 0）
  python3 term_unify_20261016.py --apply  # 写入
  python3 term_unify_20261016.py --check  # 只判定：有漂移则退出码 1（CI 用）

  两者都可加 --root <dir> 指定扫描根；默认是仓库根。生产调用不带该参数，
  它只给守卫测试植入样本用（见 tests/ao_shaping/scripts/test_term_gate_enforced.py）。
"""
import ast as _ast
import io
import re
import sys
import tokenize
from pathlib import Path

# 自定位：脚本位于 <repo>/scripts/_common/term_unify_20261016.py
# 故仓库根 = parents[2]（_common -> scripts -> repo）。
REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# 精确替换规则（顺序敏感：长匹配在前）。均为中文术语，不涉及代码标识符。
# ---------------------------------------------------------------------------
EXACT_RULES = [
    # T-01 平场泄漏
    ("平场泄漏系数", "加性背景（杂散光）项系数"),
    ("平场泄漏校准", "加性背景（杂散光）项校准"),
    ("平场泄漏", "加性背景（杂散光）项"),
    # T-20 瞳孔 → 瞳面（先于 pupil 夹注逻辑，避免误判）
    ("瞳孔", "瞳面"),
    # T-02 台架
    # 组合词（2026-10-22，issue #187 二审：真+台架 需整体映为 实验系统，
    # 与 真机→实验系统 一致；否则裸 台架 规则会映出冗余的「真实验系统」）
    ('真台架',     '实验系统'),
    ('真实台架',   '实验系统'),
    ('2f-Fourier SLM-PIB 仿真台架', '2f-Fourier SLM-PIB 仿真系统'),
    ('2f-Fourier 台架',             '2f-Fourier 实验系统'),
    ('2f 台架',                     '2f 实验系统'),
    ('台架事实',   '实验系统事实'),
    ('台架探针',   '实验系统探针'),
    ('台架纪律',   '实验系统纪律'),
    ('台架实测',   '实验系统实测'),
    ('台架稳定性', '实验系统稳定性'),
    ('台架参数',   '实验系统参数'),
    ('台架健康度', '实验系统健康度'),
    ('台架验收',   '实验系统验收'),
    ('台架表征',   '实验系统表征'),
    ('台架',       '实验系统'),

    # T-66 真实硬件（2026-10-22，issue #187 登记：此前该词同时绕过引擎与 git-grep
    # 兜底）。语义分层（同 T-11 注）：
    #   - 指"硬件实体本身"（无需/没有/模拟/连接/依赖/打开 真实硬件）→ 硬件
    #   - 指"区别于仿真 sim 的实验"（真实硬件 语料/图像/采集/标度/CLI/实测）→ 实验
    #   - 特例：真实硬件 曝光 → 实测曝光
    # 顺序敏感：组合词先于裸词。与 T-11 无子串重叠（真实硬件 不含 实机），可安全共表。
    ('真实硬件曝光', '实测曝光'),
    ('无需真实硬件', '无需硬件'),
    ('没有真实硬件', '没有硬件'),
    ('模拟真实硬件', '模拟硬件'),
    ('连接真实硬件', '连接硬件'),
    ('依赖真实硬件', '依赖硬件'),
    ('打开真实硬件', '打开硬件'),
    # 指"硬件实体本身"的其余搭配（SECURITY 安全路径 / 架构图驱动分支 / CLI 设备上下文）
    ('涉及真实硬件', '涉及硬件'),
    ('真实硬件驱动', '硬件驱动'),
    ('真实硬件，', '硬件，'),
    ('真实硬件上', '实验系统上'),
    ('真实硬件上的', '实验系统上的'),
    # 组合词先于裸词（本表既定不变量）：缺了这一条，裸词只吃掉前缀，留下重复的
    # 「实验实验」。映射照抄同表 T-03 的同类组合（…实验 → 实验），不登记新词。
    ('真实硬件实验', '实验'),
    ('真实硬件',   '实验'),

    # T-03 实机 / 真机 / 上机
    ("实机实验", "实验"),
    ("未上真机", "未完成实验验证"),
    ("真机 A/B", "实验系统 A/B"),
    ("真机首跑", "首次实验运行"),
    ("真机验证", "实验系统验证"),
    ("真机运行", "实验运行"),
    ("上真机", "上实验系统验证"),
    ("真机上", "实验系统上"),
    ("真机", "实验系统"),
    ("实机验收", "实验验收"),
    ("实机闭环", "实测闭环"),
    ("实机调试", "实验调试"),
    ("实机对照", "实验对照"),
    ("实机运行", "实验运行"),
    ("实机数据", "实验数据"),
    ("实机测量", "实验测量"),
    ("实机增益", "实验增益"),
    ("实机失败", "实验失败"),
    ("实机标定", "实验标定"),
    ("实机光路", "实验光路"),
    ("实机", "实验"),
    ("上机", "实验"),
    # T-04 口语化
    ("跑通该流程", "完整在线实现该流程"),
    ("跑通", "在线实现"),
    ("互相追逐", "相互耦合振荡"),
    ("系数范数被灌进", "系数范数转移到"),
    ("的系数范数灌进", "的系数范数转移到"),
    ("灌进", "转移到"),
    ('目标函数被"钻空子"', "目标函数被规避（表观改善而非真实优化）"),
    ("光斑被彻底打散", "光斑完全弥散"),
    ("光斑能量被显著打散", "光斑能量被显著弥散"),
    ("打散光场", "弥散光场"),
    ("被打散", "被弥散"),
    ("金标准", "参考（真值）"),
    ("围栅", "环围"),
    ("回复为", "重构为"),
    ("板卡", "控制卡"),
    ("撞满量程", "达到量程饱和"),
    ("数值对拍", "数值一致性比对"),
    ("现定曝光", "会话内在线确定的曝光"),
    ("实测甜点", "实测最优学习率"),
    ("粉饰", "拔高"),
    # 注: ('本方','本文') (T-42) 于 2026-10-20 移出本表, 改走下方 REGEX_RULES 的
    #     负向前瞻规则——裸 replace 曾把 "本方案" 误吞成 "本文案"。
    # T-61 反向修复：T-60 的裸词规则曾把 GS 语境一起吞掉并已提交入库，那些文件里已不存在
    # 「自由相位」，前向规则无法命中，术语表复查会误报 0 差异而实际语义仍是假的。
    # 故显式把已被误改的 "GS 全频随机相位" 收回 GS 语境用词。必须是独立长规则，
    # 不能靠放宽 T-60——那会连带回退探针语境里正确的「全频随机相位」。
    ("GS 全频随机相位", "GS 逐像素相位"),
    # T-61 GS 自由相位 → GS 逐像素相位（2026-10-19）。**必须先于 T-60**：这是 GS 优化
    # 得到的确定性相位（逐像素相位自由度，AGENTS.md "full-pixel phase freedom"），
    # 不是 `freeform` 随机探针；若让 T-60 的裸词规则先命中，会把可检验的断言改成假的。
    ("GS 自由相位", "GS 逐像素相位"),
    # T-60 自由相位 → 全频随机相位（2026-10-19）。代码标签 `freeform` 不在此列：
    # 它是标识符，由 §2 保护；此处只统一中文叙述。
    # 适用范围仅 `freeform` 随机探针（见术语表 T-60/T-61 说明）。
    ("自由相位留出", "全频随机相位留出"),
    ("自由相位族", "全频随机相位族"),
    ("自由相位帧", "全频随机相位帧"),
    ("自由相位两族", "全频随机相位两族"),
    ("自由相位", "全频随机相位"),
    # T-62 留出 → 验证集（2026-10-20，数据集划分术语；学术汇报 v6 先行，仓库跟进）。
    # 顺序敏感：组合词先于裸词；量词搭配与 deck v6 对齐（「N 张验证集帧」）。
    # 代码标识符（holdout_*、freeform_holdout_count、--holdout-probe-count）不在
    # 中文词表内，天然不受影响（§2 DO-NOT-TOUCH）。
    #
    # 多义保护（2026-10-22，issue #187 .py 扩展）：「留出」除 holdout 义外还有
    # "预留/腾出"义（留出时间、留出几何/空间/余量），须先占位否则被裸词吞成
    # 「验证集时间/几何」。用占位符保护后再还原（见 process_py 的 _t62_holdout_guard）。
    ('张留出帧',   '张验证集帧'),
    ('张留出',     '张验证集帧'),
    ('留出集合',   '验证集'),
    ('留出集',     '验证集'),
    ('留出',       '验证集'),

    # T-63 综合分 → 复合分数（2026-10-20，指标术语；学术汇报 v6 扫描发现 deck
    # 与报告在 composite 指标上混用「composite / 复合分数 / 综合分」三种写法）。
    # 规范统一为「复合分数」（与论文 v4.2 及 report/ 主流一致）。顺序敏感：
    # 组合词「综合分数」先于裸词，避免产出「复合分分数」。
    ("综合分数", "复合分数"),
    ("综合分", "复合分数"),

    # -----------------------------------------------------------------------
    # T-67 模型术语统一：本仓库描述「由 SLM 瞳面相位预测远场强度」的
    # 确定性物理模型（区别于 U-Net 等学习网络）一律称 **物理预测模型**。
    # 旧写法 正向模型 / 前向模型 / 物理模型 / 物理前向 / 物理正向 等
    # 混用 → 统一。修饰词 闭式 / 可微 / 可微傅里叶 保留在前。
    #
    # 顺序敏感（最长在前，与全表不变量一致）：
    #   - 「可微正向模型 / 物理正向模型 / 物理前向模型 / 闭式物理前向 /
    #     可微物理前向」(6 字) 必须先于裸词「物理前向 / 物理正向」(4 字)，
    #     否则被裸词先咬掉前缀、产出「可微物理预测 + 残留」。
    #   - 「可微正向模型」(6) 先于「正向模型」(4)：前者含后者作后缀。
    #
    # 幂等：目标串（物理预测模型 / 物理预测 / 闭式物理预测 / 可微物理预测 /
    #   可微物理预测模型）不含任何源串子串——「物理模型」要求「物理」紧跟
    #   「模型」，而「物理预测模型」中「物理」后是「预测」，不命中。
    #
    # 边界 A（U-Net 不冠「物理」）：U-Net 是学习网络，不是物理模型，故
    #   「U-Net 深度前向 / 深度学习前向路径 / U-Net 与物理模型」中的 U-Net
    #   路径不命中本表（无 物理模型/正向模型 源串覆盖它）；只有与之对照的
    #   物理模型一侧被统一。
    # 边界 B（通用「前向」保留）：前向算子 / 前向计算链 / 单次前向 / 正向收益 /
    #   正向预测 / 可微前向 / 深度前向 等是标准光学义（forward operator / pass /
    #   direction），非「模型名」，本表不收录、保留原样。
    # -----------------------------------------------------------------------
    # T-69 乱序/重复修复：T-67 裸词规则把「预测」插进 物理/正向/前向 之间后，
    #   deck 源（slides_ppt_20261023.json / 学术汇报 build 脚本）出现词序错乱
    #   「物理正向预测模型 / 正向物理预测模型 / 正向预测模型」，统一回规范名。
    #   8 字「物理正向预测模型」须先于 6 字「物理正向模型」与 4 字「物理正向」。
    #
    # 2026-10-26 补两条（T-69 二期），各成对「预防 + 回溯」：
    #   - 预防（源串 物理前向 尚存的未入库文件）：
    #       物理前向预测模型→物理预测模型（8）、物理前向预测→物理预测（7）
    #     必须**先于** 4 字裸词「物理前向」，否则 4 字先把前缀咬掉、产出
    #     「物理预测预测模型」的 预测 重复乱序（实测 experiment_summary_20261022）。
    #   - 回溯（已入库的 预测 重复乱序，源串已消失、正向规则命中不到）：
    #       物理预测预测模型→物理预测模型（8）、物理预测预测→物理预测（7）
    #     同 T-61「显式写回已误改串」思路——已写坏的串只有反向规则能收回。
    #   全部幂等：目标串（物理预测/物理预测模型）不含任何源串子串。
    ("物理前向预测模型", "物理预测模型"),   # 8 预防
    ("物理预测预测模型", "物理预测模型"),   # 8 回溯
    ("物理正向预测模型", "物理预测模型"),   # 8
    ("正向物理预测模型", "物理预测模型"),   # 8
    ("正向预测模型", "物理预测模型"),        # 6
    ("可微正向模型", "可微物理预测模型"),
    ("物理正向模型", "物理预测模型"),
    ("物理前向模型", "物理预测模型"),
    ("闭式物理前向", "闭式物理预测"),
    ("可微物理前向", "可微物理预测"),
    ("物理前向预测", "物理预测"),           # 7 预防（先于 4 字 物理前向）
    ("物理预测预测", "物理预测"),           # 7 回溯（先于 4 字 物理前向）
    ("物理前向", "物理预测"),
    ("物理正向", "物理预测"),
    ("闭式前向", "闭式物理预测"),
    # 注意：裸词「正向模型 / 前向模型」**不放本表**，改走 REGEX_RULES 的带
    #   学习侧守卫规则（见下 T-67 边界 A）：学习网络侧「深度正向模型 /
    #   端到端学习前向模型」里的 正向/前向 是标准光学义（forward），须保留，
    #   只有物理模型一侧才统一为 物理预测模型。
    ("物理模型", "物理预测模型"),

    # -----------------------------------------------------------------------
    # T-70 验收/复测协议术语（2026-10-30，用户指示：优化「重放」「逐块 A-B-B-A」）：
    # 「重放」→「复测」；A-B-B-A 复测 / 冻结重复验证（E 阶段）。
    # 注意：「交错重放」「common_replay」「重放审计」为独立协议名，保留不改（见术语表 §1.8）。
    ("A-B-B-A 重放", "A-B-B-A 复测"),
    ("A-B-B-A 逐块验收", "A-B-B-A 逐块复测验收"),
    ("冻结重放", "冻结重复验证"),
    # T-68 调优方法术语统一：本仓库两类「在线调优方法」的规范名：
    #   (1) 有预训练部分（源域预训练 + 少量探针迁移到新光路）→ **迁移学习**
    #   (2) 从零在线重拟合（model-in-loop，无预训练基座）→ **MIL / model-in-loop**
    # 旧变体：迁移模型（把 (1) 的「迁移学习」方法误写成「迁移模型」）→ 迁移学习。
    # 幂等：目标串「迁移学习」不含源串「迁移模型」子串。
    # 不动：迁移（裸词，语义多为 文件/代码/参数 迁移，噪声大，统一会破坏
    #   「迁移学习」→「迁移学习学习」）、微调（fine-tuning 子步骤，合法独立
    #   术语）、物理迁移 / 参数迁移（transfer 工作流内的**子步骤**名，非方法名）、
    #   model-in-loop / MIL（(2) 的规范名，已占多数 128 处）。
    # -----------------------------------------------------------------------
    ("迁移模型", "迁移学习"),
]

# ---------------------------------------------------------------------------
# T-65 「臂」→「变体」（2026-10-22，A/B 对照语境）
#
# 规范词取 ML / MLOps 的专业术语 **variant（变体）**：同一个系统的两条 A/B 对照
# 配置在机器学习语境里就叫 variant。仓库语境：model-in-loop A/B campaign 的
# `baseline_a` / `baseline_b` / `large_1000`、仿真方法的 `zernike_only` /
# `perpixel_only` / `joint`、前向模型架构的 `conv` / `mlp` —— 都是同一命令的
# 两种（多种）门槛/配置设定。
#
# ⚠️ **例外（必须先于裸词规则）**：OOPAO 对比语境里的「臂」指的是 **numpy 路径与
# OOPAO 路径这两个并列的仿真后端分支**，不是 A/B 对照的臂 → 一律写
# 「**两个仿真后端分支** / **仿真后端分支**」（同文件本来就把它们叫 backend /
# 后端）。**为什么不用「双后端」**：那两个是**两个并列的仿真后端分支**（两套独立
# 实现），**不是同一后端的两种配置**；叫「双后端」会被读成后者。见术语表 §1.9。
#
# **规则顺序约定**（同 T-61「GS 语境先于裸词」）：
#   1. 数量词组合先于裸词（「两臂」→「两个变体」，否则会产出「两变体」这种
#      半文半白的写法；同理「A 臂」→「变体 A」）。
#   2. 「双臂」例外先于裸词，否则会被吞成「双变体」。
#
# 代码标识符（`baseline_a` / `baseline_b` / `large_1000` / `arm` / `arms` /
# `ARM_LABEL` / `backend_arm`、stage 名、CLI 选项名）**一律不改**（§2 保护）：
# 本表只匹配中文叙述里的「臂」字。
# ---------------------------------------------------------------------------
T65_VARIANT_RULES = [
    # ⚠️ 例外先行：「双臂」在仓库里的唯一含义是 numpy / OOPAO 两个**仿真后端分支**
    #    （见术语表 §1.9），不能被下面的裸词规则吞成「双变体」。
    ("双臂", "两个仿真后端分支"),
    ("两臂", "两个变体"),
    ("三臂", "三个变体"),
    ("各臂", "各变体"),
    ("每臂", "每个变体"),
    ("单臂", "单个变体"),
    ("两条臂", "两个变体"),
    ("第二条臂", "第二个变体"),
    ("第一条臂", "第一个变体"),
    ("一条臂", "一个变体"),
    ("一臂", "一个变体"),
    ("3 臂", "3 个变体"),
    ("2 臂", "2 个变体"),
    ("臂", "变体"),
]

# 「A 臂」/「B 臂」→「变体 A」/「变体 B」。
# 这条必须走正则而不是字符串替换：源文「B 臂把门槛…」里「臂」与后文之间**没有**
# 空格，纯 replace 会产出半文半白的「变体 B把门槛…」。故按后文首字符决定补不补空格
# （后文是汉字则补，是括号/标点则不补）。
T65_AB_ARM_RE = re.compile(r"([AB]) 臂")


def _t65_ab_arm(m):
    tail = m.string[m.end() : m.end() + 1]
    sep = " " if tail and re.match(r"[\u4e00-\u9fff]", tail) else ""
    return f"变体 {m.group(1)}{sep}"


# T-65 例外：这些文件里「臂」= numpy / OOPAO 两个**并列的仿真后端分支**，
# 不是 A/B 变体。这些文件写「两个仿真后端分支 / 仿真后端分支」；其余文件写「变体」。
# ⚠️ 「双臂」→「两个仿真后端分支」在两套规则里都放在第一条：它是全局例外，任何
#    文件里的「双臂」都不允许被裸词规则吞成「双变体」。
#
# 表内后几条（「双后端 / 两后端 / 两个后端 / 每条后端 / 某后端 / 任一个后端 /
# 2 条后端」）是**反向守卫**：万一有人把旧措辞写回来，直接在这里改写成新措辞。
# 它们全部幂等 —— 产物「两个仿真后端分支」不含任何源串，不会被再次命中。
# ⚠️ 这张表**只对 T65_BACKEND_FILES 里的文件生效**（文件级例外）：仓库别处的
#    「两个后端」说的是别的东西（如 ccd 驱动里 Daheng / MiiCam 两个相机后端），
#    绝不能被这张表误伤。
T65_BACKEND_FILES = (
    "report/oopao_impact/oopao_impact_report/report.md",
    "report/oopao_vs_numpy/oopao_vs_numpy_report/report.md",
    "scripts/oopao/generate_oopao_impact_report.py",
    "scripts/oopao/generate_oopao_vs_numpy_report.py",
    "src/ao_shaping/drivers/sim/AGENTS.md",
    "src/ao_shaping/drivers/sim/beam_backend.py",
)

T65_BACKEND_RULES = [
    ("两后臂", "两个仿真后端分支"),
    ("后端臂", "仿真后端分支"),
    ("双臂", "两个仿真后端分支"),
    ("两臂", "两个仿真后端分支"),
    ("两条臂", "两个仿真后端分支"),
    ("第二条臂", "第二个仿真后端分支"),
    ("第一条臂", "第一个仿真后端分支"),
    ("一条臂", "一个仿真后端分支"),
    ("单臂", "单个仿真后端分支"),
    ("每臂", "每个仿真后端分支"),
    ("一臂", "一个仿真后端分支"),
    ("臂", "仿真后端分支"),
    # --- 旧措辞 → 新措辞（幂等反向守卫，见上） ---
    ("两个后端", "两个仿真后端分支"),
    ("双后端", "两个仿真后端分支"),
    ("两后端", "两个仿真后端分支"),
    ("2 个后端", "2 个仿真后端分支"),
    ("2 条后端", "2 个仿真后端分支"),
    ("每条后端", "每个仿真后端分支"),
    ("某后端", "某个仿真后端分支"),
    ("任一个后端", "任一仿真后端分支"),
]

# ---------------------------------------------------------------------------
# 正则级术语规则（带上下文守卫）：用于裸子串会吞噬合成词的特例。
# ---------------------------------------------------------------------------
# T-67 边界 A（学习侧守卫，2026-10-26 三期：上下文窗口版）先定义后引用：
# REGEX_RULES 在模块级构造，引用的守卫与 token 表必须在此之前（否则导入即
# NameError）。
#
# 裸词「正向模型 / 前向模型」（无 物理 前缀）默认是物理侧 → 物理预测模型；
# 但若处于学习网络侧（U-Net / GSNet / 深度 / 端到端学习 / hybrid / 学习型
# 前向 等），正向/前向 是标准光学义（forward，学习网络的前向路径），不冠
# 「物理」。负向后行断言 (?<!深度)(?<!学习)(?<!型) 在 **加粗** / 拉丁 token
# / 型 字隔开的情况下都会漏判：v6 deck「GSNet 学习型前向模型」被误转成
# 「…物理预测模型」、zernike「**学习出来的**前向模型」同样漏判。改用 12 字
# 上下文窗口（同 pupil / bake-off 的窗口判据），窗口内出现学习侧 token 即
# 保留原串；物理侧（无学习侧 token）照常统一。物理前缀的 物理正向/物理前向
# 已由 EXACT_RULES 先行命中，不会走到这里。
# 学习侧 token（12 字窗口内出现即判学习侧）：中文 token 为主（U-Net/GSNet
# 等拉丁名总与 深度/学习/黑盒 同现）；Net 兜底防拉丁名被窗口截断。物理侧
# 描述词（物理/可微/闭式/FFT/冻结）不在此列，照常统一。
_LEARNING_SIDE_TOKENS = ("学习", "深度", "端到端", "黑盒", "卷积",
                         "hybrid", "U-Net", "GSNet", "CNN", "Net")
_FORWARD_MODEL_RE = re.compile(r"(?<!物理)(?:正向|前向)模型")


def _forward_model_guard(m):
    """T-67 边界 A：学习侧「正向/前向 模型」保留，物理侧统一为 物理预测模型。"""
    i = m.start()
    window = m.string[max(0, i - 12):i]
    if any(tok in window for tok in _LEARNING_SIDE_TOKENS):
        return m.group(0)  # 学习侧：保留 正向 / 前向
    return "物理预测模型"  # 物理侧：统一


REGEX_RULES = [
    # T-42 本方→本文（"我方"义, 如 "本方立场"）: 负向前瞻保住 "本方案/本方向/
    # 本方式/本方法/本方针/本方块/本方阵/本方格" 等合成词, 不再误吞。
    (re.compile(r"本方(?!案|向|式|法|针|块|阵|格)"), "本文"),
    # T-64 均匀性→均匀度（2026-10-20，uniformity 语境，对齐 8 篇 CrossRef 对标文献；
    # 仓库多数派 101 vs 84，deck 自身 S3 与 S24/S31/S35 自相矛盾）。
    # 负向后行断言排除 "不均匀性"（物理性质描述，如 "高频不均匀性"）；
    # 负向前瞻排除指标标识符 "均匀性_CV"（同 uniformity_cv，数据字段名，§2 保护）。
    (re.compile(r"(?<!不)均匀性(?!_)"), "均匀度"),
    # T-67 边界 A 学习侧守卫（窗口版，见上方 _forward_model_guard 注释）。
    (_FORWARD_MODEL_RE, _forward_model_guard),
]


# ---------------------------------------------------------------------------
# 代码/围栏检测
# ---------------------------------------------------------------------------
def _in_inline_code(line, pos):
    """pos 之前本行反引号为奇数个 → 处于行内 `code` 中"""
    seg = line[:pos]
    return seg.count("`") % 2 == 1


def _ascii_dominant(s):
    if not s.strip():
        return False
    cjk = len(re.findall(r"[\u4e00-\u9fff]", s))
    asc = len(re.findall(r"[A-Za-z0-9]", s))
    return asc >= 3 and cjk == 0


def _has_cjk(s):
    return bool(re.search(r"[\u4e00-\u9fff]", s))


# ---------------------------------------------------------------------------
# 登记记录行保护
# ---------------------------------------------------------------------------
# 这些行把禁词作为「被替换对象」的证据来引用（登记标题、对照表行、"某词是 code 标签
# 的直译"这类说明）。替换它们会摧毁记录本身，并使本脚本非幂等：第二次运行会把
# 「T-60：自由相位 → 全频随机相位」改写成「T-60：全频随机相位 → 全频随机相位」。
RECORD_LINE_PATTERNS = (
    re.compile(r"术语登记"),
    re.compile(r"^\s*[|>#+-]*\s*T-\d+\b"),
    re.compile(r"是代码标签\s*`"),
)


def _is_registration_record(line):
    return any(p.search(line) for p in RECORD_LINE_PATTERNS)


# ---------------------------------------------------------------------------
# 单行处理
# ---------------------------------------------------------------------------
def process_line(line, t65_rules=T65_VARIANT_RULES):
    out = line

    # 1) 精确术语
    for old, new in EXACT_RULES:
        if old in out:
            out = out.replace(old, new)

    # 1.5) 正则级术语（带上下文守卫, 见 REGEX_RULES 注释）
    for pat, new in REGEX_RULES:
        out = pat.sub(new, out)

    # 1.6) T-65 「臂」→ 变体 / 后端（文件级例外，见 T65_* 注释）。
    #      放在 EXACT_RULES 之后、pupil/bake-off 之前：数量词组合与「双臂」例外
    #      都在同一张表里按序生效，不受其它规则干扰。
    #      「A 臂 / B 臂」走正则（_t65_ab_arm），必须先于裸词规则。
    out = T65_AB_ARM_RE.sub(_t65_ab_arm, out)
    for old, new in t65_rules:
        if old in out:
            out = out.replace(old, new)

    # 2) 英文 pupil：仅中文叙述中加注释；代码符号/英文叙述/行内 code 保留
    line_has_pupil_cn = ("瞳面" in out) or ("瞳孔" in out)

    def pupil_repl(m):
        i = m.start()
        if _in_inline_code(out, i):
            return m.group(0)
        if i + 5 < len(out) and re.match(r"[A-Za-z0-9_]", out[i + 5]):
            return m.group(0)
        if i > 0 and re.match(r"[A-Za-z0-9_]", out[i - 1]):
            return m.group(0)
        if line_has_pupil_cn:
            return m.group(0)
        # 判据窗口必须止于本行。跨行窗口会让"英文句子换行后紧跟中文段落"的样本
        # 被中文污染, 把纯英文的 pupil 误判成中文叙述并加夹注 (实测把
        # scripts/README.md 的 `(focal scale, pupil` 改成了 `pupil（瞳面）`,
        # 而术语表只要求**中文叙述**首现加注)。同窗口内判据才与该口径一致。
        w = 40
        line_start = out.rfind("\n", 0, i) + 1
        line_end = out.find("\n", i)
        if line_end < 0:
            line_end = len(out)
        before = out[max(line_start, i - w) : i]
        after = out[i + len("pupil") : min(line_end, i + len("pupil") + w)]
        after_is_ascii = not after.strip() or _ascii_dominant(after)
        if _ascii_dominant(before) and after_is_ascii and not _has_cjk(before + after):
            return m.group(0)
        return "pupil（瞳面）"

    out = re.sub(r"\bpupil\b", pupil_repl, out, flags=re.IGNORECASE)

    # 3) bake-off → 对比择优（2026-10-17 起全中文化，T-54）：
    #    先处理带夹注形式 bake-off（对比择优），再处理裸词；
    #    英文叙述/行内 code 保留（代码标识符 bake_off_rejected 用下划线，不受影响）。
    def _bakeoff_ctx_keep(m):
        i = m.start()
        if _in_inline_code(out, i):
            return m.group(0)
        w = 40
        before = out[max(0, i - w) : i]
        after = out[i + len("bake-off") : i + len("bake-off") + w]
        if not (_has_cjk(before) or _has_cjk(after)):
            return m.group(0)
        return "对比择优"

    out = re.sub(r"\bbake-off\b（对比择优）", _bakeoff_ctx_keep, out)
    out = re.sub(r"\bbake-off\b", _bakeoff_ctx_keep, out)

    return out


def transform(text, rel_path=""):
    """rel_path: 仓库相对路径（posix 风格）。用于 T-65 的按文件例外。"""
    # T-65 文件级例外：OOPAO 语境写「仿真后端分支」，其余文件写「变体」。
    t65_rules = (
        T65_BACKEND_RULES if rel_path in T65_BACKEND_FILES else T65_VARIANT_RULES
    )
    lines = text.split("\n")
    fence = False
    out = []
    for ln in lines:
        if ln.strip().startswith("```"):
            fence = not fence
            out.append(ln)
            continue
        if fence:
            out.append(ln)
            continue
        if _is_registration_record(ln):
            out.append(ln)
            continue
        out.append(process_line(ln, t65_rules))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# .py 处理管线（2026-10-22，issue #187）
# ---------------------------------------------------------------------------
# 只处理 tokenizer 识别出的 STRING / COMMENT token 内的中文叙述；代码标识符、
# 围栏代码、纯英文（§2）天然不受影响。pupil / bake-off 英文夹注规则不适用。
# f-string 的 {expr} 洞保护（表达式内可能是与禁词同名的标识符/字典键）。
#
# .py 排除集：
#   - 本文件自身：规则表用禁词作源串，扫描会自我改写、失去幂等（同 .md 侧
#     EXCLUDE_NAMES 排除术语表/AGENTS.md 的理由）。
#   - 禁词守卫测试：banned 元组是"断言正文不得含禁词"的测试数据，替换即反转守卫。
#   - libs/：vendored 第三方库，不属于本仓库叙述。
#   - call_graph_wiki：生成文档。
PY_EXCLUDE = {
    'scripts/_common/term_unify_20261016.py',
    'tests/ao_shaping/runners/test_model_in_loop_session_preflight.py',
    # 这个文件的注释与字符串刻意保留禁词作为**测试夹具**(门禁必须能证明它会报错)。
    # 改写它等于把"检测器会失败"的测试改成"永远通过"。
    'tests/ao_shaping/scripts/test_term_gate_enforced.py',
}

# 「留出」多义保护：holdout 义之外的 "预留/腾出" 义（留出时间/几何/空间/余量），
# 先占位再让 EXACT 规则跑，跑完还原（否则裸词规则吞成「验证集时间/几何」）。
_HOLDOUT_SAFE_RE = re.compile(r'留出\s*(?=时间|几何|空间|余量|余地|缺口|缓冲|间隔)')

def _is_py_meta_line(line):
    """术语元注释行：引用禁词作"不用 X"式反例（同 .md 侧登记记录行保护）。"""
    return '不用「' in line

def _apply_cn_rules(seg, t65_rules):
    """对一段 .py 字符串/注释文本应用中文术语规则（不含 pupil/bake-off）。"""
    out = seg
    for old, new in EXACT_RULES:
        if old in out:
            out = out.replace(old, new)
    for pat, new in REGEX_RULES:
        out = pat.sub(new, out)
    out = T65_AB_ARM_RE.sub(_t65_ab_arm, out)
    for old, new in t65_rules:
        if old in out:
            out = out.replace(old, new)
    return out

def _fstring_holes(seg):
    """f-string 段内 {expr} 洞的 (start, end) 区间列表（相对 seg 起点）。"""
    holes = []
    depth = 0
    hstart = None
    for j, ch in enumerate(seg):
        if ch == '{':
            if depth == 0:
                hstart = j
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and hstart is not None:
                if not (hstart + 1 < len(seg) and seg[hstart + 1] == '{'):  # {{ 转义
                    holes.append((hstart, j + 1))
                hstart = None
    return holes

def transform_py(text, rel_path=''):
    """对 .py 的字符串字面量与注释应用中文术语规则（.py 侧不含 pupil/bake-off）。
    只处理 tokenizer 识别出的 STRING/COMMENT token；代码标识符、纯英文（§2）
    天然不受影响。f-string 的 {expr} 洞与「留出」多义组合词先保护后还原。"""
    t65_rules = T65_BACKEND_RULES if rel_path in T65_BACKEND_FILES else T65_VARIANT_RULES
    lines = text.split('\n')
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (SyntaxError, tokenize.TokenError, IndentationError):
        return text
    # 行 -> [(start, end, is_fstring)]
    # STRING: 普通字符串（f-string 整体在 <3.12 是一个 STRING token，需洞保护）。
    # FSTRING_MIDDLE: 3.12+ 多行 f-string 的**纯文本**段（{expr} 已拆成
    #   FSTRING_PART 独立 token），无需洞保护，直接按普通文本处理。
    #   缺这类 token 会漏掉报告生成器里 f-string 模板的中文叙述（issue #187 主战场）。
    spans = {}
    for tok in toks:
        if tok.type in (tokenize.STRING, tokenize.COMMENT):
            is_f = tok.string[:1] in 'fF' and len(tok.string) > 1 and tok.string[1] in '"\''
        elif tok.type == getattr(tokenize, 'FSTRING_MIDDLE', None):
            is_f = False  # 表达式已分离，文本段按普通字符串处理
        else:
            continue
        sl, sc = tok.start[0] - 1, tok.start[1]
        el, ec = tok.end[0] - 1, tok.end[1]
        if sl == el:
            spans.setdefault(sl, []).append((sc, ec, is_f))
        else:
            # 多行字符串（docstring / f-string 模板）：按行拆段
            spans.setdefault(sl, []).append((sc, len(lines[sl]), is_f))
            for i in range(sl + 1, el):
                spans.setdefault(i, []).append((0, len(lines[i]), is_f))
            spans.setdefault(el, []).append((0, ec, is_f))
    for li, segs in spans.items():
        line = lines[li]
        if _is_py_meta_line(line):
            continue
        for (s, e, is_f) in segs:
            seg = line[s:e]
            if not _has_cjk(seg):
                continue
            newseg = _apply_py_segment(seg, t65_rules, is_f)
            if newseg != seg:
                line = line[:s] + newseg + line[e:]
        lines[li] = line
    return '\n'.join(lines)

_HOLE_PH = '\x01'     # f-string {expr} 洞占位
_HOLDOUT_PH = '\x02'  # 「留出」多义组合词占位

def _protect_holes(seg, is_f):
    """把 f-string {expr} 洞替换为占位符，返回 (新串, 洞文本列表)。"""
    if not is_f:
        return seg, []
    holes = _fstring_holes(seg)
    if not holes:
        return seg, []
    parts, saved = [], []
    pos = 0
    for (hs, he) in holes:
        parts.append(seg[pos:hs])
        saved.append(seg[hs:he])
        parts.append(_HOLE_PH)
        pos = he
    parts.append(seg[pos:])
    return ''.join(parts), saved

def _restore_holes(seg, saved):
    idx = [0]
    def _ph(m):
        s = saved[idx[0]] if idx[0] < len(saved) else ''
        idx[0] += 1
        return s
    return re.sub(_HOLE_PH, _ph, seg)

def _apply_py_segment(seg, t65_rules, is_f):
    """单段 .py 字符串/注释的术语替换（含 f-string 洞与 留出 多义保护）。"""
    # 1) f-string {expr} 洞先保护（洞内可能是与禁词同名的标识符/字典键）
    seg, hole_saved = _protect_holes(seg, is_f)
    # 2) 「留出」多义组合词（留出时间/几何…）占位，防裸词规则吞成「验证集…」
    holdout_saved = []
    def _hold_ph(m):
        holdout_saved.append(m.group(0))
        return _HOLDOUT_PH
    seg = _HOLDOUT_SAFE_RE.sub(_hold_ph, seg)
    # 3) 应用中文术语规则
    seg = _apply_cn_rules(seg, t65_rules)
    # 4) 还原占位（先 留出 后 洞）
    idx = [0]
    def _hold_restore(m):
        s = holdout_saved[idx[0]] if idx[0] < len(holdout_saved) else ''
        idx[0] += 1
        return s
    seg = re.sub(_HOLDOUT_PH, _hold_restore, seg)
    return _restore_holes(seg, hole_saved)

# ---------------------------------------------------------------------------
# 文件清单。论文走独立工作流；术语表与 agent 配置"故意"引用禁词作示例，须排除。
# ---------------------------------------------------------------------------
EXCLUDE_NAMES = {
    "terminology_glossary.md",  # 术语表：禁词作示例，须排除
    "AGENTS.md",  # 术语规则块引用禁词作示例
    "CLAUDE.md",  # 同上（.claude/）
    # 论文：术语表同样适用；repo 的 v3.x 副本与工作区 v4.0 一并统一
    # （工作区 v4.0 已在上轮润色，此处为 no-op）
}
#: 机内本地目录与运行时产物：不属于仓库文档，与根 ``.gitignore`` 的
#: machine-local 清单保持一致。它们不进版本控制，术语门禁不该被它们拖成
#: 永不收敛（2026-10-22: ``.workbuddy/memory/*.md`` 的工作日志把 dry-run
#: 从 0 个文件变成 1 个文件；2026-10-23: ``.history/`` 的 IDE 快照同理）。
LOCAL_DIR_PREFIXES = (
    ".workbuddy/",
    ".claude/",
    ".codegraph/",
    ".slim/",
    ".streamlit/",
    ".auto_doc_review/",
    "_scripts/",
    "logs/",
    ".history/",
)

def main(argv: "list[str] | None" = None) -> int:
    """Scan --root and report drift; import-time work is exactly nothing.

    The scan used to run at module scope straight off ``sys.argv``, so every importer
    inherited a full-repo walk (about 4 s here) plus stdout, and inherited write mode
    if the ambient argv happened to contain --apply. tests/ao_shaping/scripts/
    test_term_gate_enforced.py records a real instance of that family: an --apply run
    against a temp --root rewrote the live worktree because the *.py pass resolved
    REPO rather than the requested root.
    """
    argv = list(sys.argv if argv is None else argv)
    apply_mode = "--apply" in argv
    check_mode = "--check" in argv
    if apply_mode and check_mode:
        raise SystemExit("--apply 与 --check 互斥: 前者写文件, 后者只判定退出码")

    #: 扫描根。默认仓库根; --root 只给守卫测试指向临时样本目录, 使"检测器真的会
    #: 报错"可被断言而不是只能相信 dry-run 输出。
    root = REPO
    for _i, _a in enumerate(argv):
        if _a == "--root" and _i + 1 < len(argv):
            root = Path(argv[_i + 1]).resolve()

    files = []
    for f in root.rglob("*.md"):
        rel = f.relative_to(root).as_posix()
        if rel.startswith((".git/", ".venv/", "venv/")):
            continue
        if rel.startswith(LOCAL_DIR_PREFIXES):
            continue
        if "call_graph_wiki" in rel:  # 生成文档，pupil 为代码标识符
            continue
        if f.name in EXCLUDE_NAMES:
            continue
        files.append(f)

    # .py 文件清单（2026-10-22，issue #187）：中文叙述（docstring/字符串/注释）
    # 进入引擎；排除集见 PY_EXCLUDE 注释。
    py_files = []
    for f in root.rglob('*.py'):
        rel = f.relative_to(root).as_posix()
        if rel.startswith(('.git', '.venv', 'venv')):
            continue
        if rel.startswith(LOCAL_DIR_PREFIXES):
            continue
        if rel.startswith('libs/') or 'libs/' in rel:   # vendored 第三方库
            continue
        if 'call_graph_wiki' in rel:
            continue
        if rel in PY_EXCLUDE:
            continue
        py_files.append(f)

    total: dict[str, int] = {}
    for f in files:
        rel_posix = f.relative_to(root).as_posix()
        orig = f.read_text(encoding="utf-8")
        new = transform(orig, rel_posix)
        if new == orig:
            continue
        changed = sum(1 for a, b in zip(orig.split("\n"), new.split("\n")) if a != b)
        total[rel_posix] = changed
        if apply_mode:
            f.write_text(new, encoding="utf-8")

    for f in py_files:
        rel_posix = f.relative_to(root).as_posix()
        orig = f.read_text(encoding='utf-8')
        new = transform_py(orig, rel_posix)
        if new == orig:
            continue
        changed = sum(1 for a, b in zip(orig.split('\n'), new.split('\n')) if a != b)
        total[rel_posix] = changed
        if apply_mode:
            f.write_text(new, encoding='utf-8')

    # stdout, on purpose: ci.yml invokes this gate with bare ``python3`` and installs
    # nothing (stdlib-only by design), and the guard tests read these lines back from
    # stdout. A logging library would move them to stderr and crash the job on import.
    tag = '文件 (dry-run)' if not apply_mode else '文件 (已写入)'
    print(f"{tag}: 变化行数")
    for k, v in sorted(total.items()):
        print(f"  {v:4d}  {k}")
    print(f"合计: {len(total)} 个文件, {sum(total.values())} 行")

    if check_mode and total:
        print("术语门禁未通过: 先跑 --apply 统一, 或改写正文后重试")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
