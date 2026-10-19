# -*- coding: utf-8 -*-
"""AO-shaping 术语统一引擎（2026-10-16）

依据 docs/terminology_glossary.md 的术语对照表，对仓库 markdown 文档做专业化替换。
规则：先特殊（长/组合）后一般（短），围栏代码块与 mermaid 图跳过，行内 `code` 保护。

用法：
  python3 term_unify_20261016.py          # dry-run（默认）
  python3 term_unify_20261016.py --apply  # 写入
"""
import re
import sys
from pathlib import Path

# 自定位：脚本位于 <repo>/scripts/_common/term_unify_20261016.py
# 故仓库根 = parents[2]（_common -> scripts -> repo）。
REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# 精确替换规则（顺序敏感：长匹配在前）。均为中文术语，不涉及代码标识符。
# ---------------------------------------------------------------------------
EXACT_RULES = [
    # T-01 平场泄漏
    ('平场泄漏系数', '加性背景（杂散光）项系数'),
    ('平场泄漏校准', '加性背景（杂散光）项校准'),
    ('平场泄漏',     '加性背景（杂散光）项'),

    # T-20 瞳孔 → 瞳面（先于 pupil 夹注逻辑，避免误判）
    ('瞳孔', '瞳面'),

    # T-02 台架
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

    # T-03 实机 / 真机 / 上机
    ('实机实验',     '实验'),
    ('未上真机',     '未完成实验验证'),
    ('真机 A/B',     '实验系统 A/B'),
    ('真机首跑',     '首次实验运行'),
    ('真机验证',     '实验系统验证'),
    ('真机运行',     '实验运行'),
    ('上真机',       '上实验系统验证'),
    ('真机上',       '实验系统上'),
    ('真机',         '实验系统'),
    ('实机验收',     '实验验收'),
    ('实机闭环',     '实测闭环'),
    ('实机调试',     '实验调试'),
    ('实机对照',     '实验对照'),
    ('实机运行',     '实验运行'),
    ('实机数据',     '实验数据'),
    ('实机测量',     '实验测量'),
    ('实机增益',     '实验增益'),
    ('实机失败',     '实验失败'),
    ('实机标定',     '实验标定'),
    ('实机光路',     '实验光路'),
    ('实机',         '实验'),
    ('上机',         '实验'),

    # T-04 口语化
    ('跑通该流程', '完整在线实现该流程'),
    ('跑通',       '在线实现'),
    ('互相追逐',   '相互耦合振荡'),
    ('系数范数被灌进', '系数范数转移到'),
    ('的系数范数灌进', '的系数范数转移到'),
    ('灌进',       '转移到'),
    ('目标函数被"钻空子"', '目标函数被规避（表观改善而非真实优化）'),
    ('光斑被彻底打散', '光斑完全弥散'),
    ('光斑能量被显著打散', '光斑能量被显著弥散'),
    ('打散光场',   '弥散光场'),
    ('被打散',     '被弥散'),
    ('金标准',     '参考（真值）'),
    ('围栅',       '环围'),
    ('回复为',     '重构为'),
    ('板卡',       '控制卡'),
    ('撞满量程',   '达到量程饱和'),
    ('数值对拍',   '数值一致性比对'),
    ('现定曝光',   '会话内在线确定的曝光'),
    ('实测甜点',   '实测最优学习率'),
    ('粉饰',       '拔高'),
    ('本方',       '本文'),
]

# ---------------------------------------------------------------------------
# 代码/围栏检测
# ---------------------------------------------------------------------------
def _in_inline_code(line, pos):
    """pos 之前本行反引号为奇数个 → 处于行内 `code` 中"""
    seg = line[:pos]
    return seg.count('`') % 2 == 1

def _ascii_dominant(s):
    if not s.strip():
        return False
    cjk = len(re.findall(r'[\u4e00-\u9fff]', s))
    asc = len(re.findall(r'[A-Za-z0-9]', s))
    return asc >= 3 and cjk == 0

def _has_cjk(s):
    return bool(re.search(r'[\u4e00-\u9fff]', s))

# ---------------------------------------------------------------------------
# 单行处理
# ---------------------------------------------------------------------------
def process_line(line):
    out = line

    # 1) 精确术语
    for old, new in EXACT_RULES:
        if old in out:
            out = out.replace(old, new)

    # 2) 英文 pupil：仅中文叙述中加注释；代码符号/英文叙述/行内 code 保留
    line_has_pupil_cn = ('瞳面' in out) or ('瞳孔' in out)
    def pupil_repl(m):
        i = m.start()
        if _in_inline_code(out, i):
            return m.group(0)
        if i + 5 < len(out) and re.match(r'[A-Za-z0-9_]', out[i+5]):
            return m.group(0)
        if i > 0 and re.match(r'[A-Za-z0-9_]', out[i-1]):
            return m.group(0)
        if line_has_pupil_cn:
            return m.group(0)
        w = 40
        before = out[max(0, i-w):i]
        after  = out[i+len('pupil'):i+len('pupil')+w]
        if _ascii_dominant(before) and _ascii_dominant(after) and not _has_cjk(before+after):
            return m.group(0)
        return 'pupil（瞳面）'
    out = re.sub(r'\bpupil\b', pupil_repl, out, flags=re.IGNORECASE)

    # 3) bake-off → 对比择优（2026-10-17 起全中文化，T-54）：
    #    先处理带夹注形式 bake-off（对比择优），再处理裸词；
    #    英文叙述/行内 code 保留（代码标识符 bake_off_rejected 用下划线，不受影响）。
    def _bakeoff_ctx_keep(m):
        i = m.start()
        if _in_inline_code(out, i):
            return m.group(0)
        w = 40
        before = out[max(0, i-w):i]
        after  = out[i+len('bake-off'):i+len('bake-off')+w]
        if not (_has_cjk(before) or _has_cjk(after)):
            return m.group(0)
        return '对比择优'
    out = re.sub(r'\bbake-off\b（对比择优）', _bakeoff_ctx_keep, out)
    out = re.sub(r'\bbake-off\b', _bakeoff_ctx_keep, out)

    return out

def transform(text):
    lines = text.split('\n')
    fence = False
    out = []
    for ln in lines:
        if ln.strip().startswith('```'):
            fence = not fence
            out.append(ln)
            continue
        if fence:
            out.append(ln)
            continue
        out.append(process_line(ln))
    return '\n'.join(out)

# ---------------------------------------------------------------------------
# 文件清单。论文走独立工作流；术语表与 agent 配置"故意"引用禁词作示例，须排除。
# ---------------------------------------------------------------------------
EXCLUDE_NAMES = {
    'terminology_glossary.md',                          # 术语表：禁词作示例，须排除
    'AGENTS.md',                                        # 术语规则块引用禁词作示例
    'CLAUDE.md',                                        # 同上（.claude/）
    # 论文：术语表同样适用；repo 的 v3.x 副本与工作区 v4.0 一并统一
    # （工作区 v4.0 已在上轮润色，此处为 no-op）
}
FILES = []
for f in REPO.rglob('*.md'):
    rel = str(f.relative_to(REPO))
    if rel.startswith(('.git', '.venv', 'venv')):
        continue
    if 'call_graph_wiki' in rel:        # 生成文档，pupil 为代码标识符
        continue
    if f.name in EXCLUDE_NAMES:
        continue
    FILES.append(f)

apply_mode = '--apply' in sys.argv
total = {}
for f in FILES:
    orig = f.read_text(encoding='utf-8')
    new = transform(orig)
    if new == orig:
        continue
    changed = sum(1 for a, b in zip(orig.split('\n'), new.split('\n')) if a != b)
    total[str(f.relative_to(REPO))] = changed
    if apply_mode:
        f.write_text(new, encoding='utf-8')

tag = '文件 (dry-run)' if not apply_mode else '文件 (已写入)'
print(f"{tag}: 变化行数")
for k, v in sorted(total.items()):
    print(f"  {v:4d}  {k}")
print(f"合计: {len(total)} 个文件, {sum(total.values())} 行")
