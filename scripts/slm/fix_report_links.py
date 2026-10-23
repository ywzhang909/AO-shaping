import re

REPORT = "D:/Projects/TIFO/AO-shaping/report/shape_trials/gs_refine/gs_issue_live_20261015.md"
with open(REPORT, "r", encoding="utf-8") as f:
    content = f.read()

pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+\.png)\)")
matches = pattern.findall(text)
print("找到的图片引用:")
for label, href in matches:
    print(f"  {label}: {href}")
print()

old_to_new = {
    # 相位图
    "../figures/gs_issue_live_20261015_205654_phase.png": "gs_issue_live_phase_20261015_205654.png",
    "../figures/gs_issue_live_20261015_205848.png": "gs_issue_live_phase_20261015_205848.png",
    "../figures/gs_issue_live_20261015_210048.png": "gs_issue_live_phase_20261015_210048.png",
    "../figures/gs_issue_live_20261015_210155.png": "gs_issue_live_phase_20261015_210155.png",
    "../figures/gs_issue_live_20261015_211345.png": "gs_issue_live_phase_20261015_211345.png",
    # 远场图：已经是正确的名字
    "../figures/gs_issue_live_farfield_20261015_205654.png": "gs_issue_live_farfield_20261015_205654.png",
    "../figures/gs_issue_live_farfield_20261015_205848.png": "gs_issue_live_farfield_20261015_205848.png",
    "../figures/gs_issue_live_farfield_20261015_210048.png": "gs_issue_live_farfield_20261015_210048.png",
    "../figures/gs_issue_live_farfield_20261015_210155.png": "gs_issue_live_farfield_20261015_210155.png",
    "../figures/gs_issue_live_farfield_20261015_211345.png": "gs_issue_live_farfield_20261015_211345.png",
}

count = 0
for old, new in old_to_new.items():
    old_link = f"]({old})"
    new_link = f"]({new})"
    count_before = text.count(old_link)
    if count_before > 0:
        text = text.replace(old_link, new_link)
        count += count_before
        print(f"  替换 {count_before} 次: {old} -> {new}")

with open(REPORT, "w", encoding="utf-8") as f:
    f.write(text)

print(f"\n总共修复了 {count} 处图片引用")
