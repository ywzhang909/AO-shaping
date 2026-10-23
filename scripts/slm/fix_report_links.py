import re

REPORT = "D:/Projects/TIFO/AO-shaping/report/shape_trials/gs_refine/gs_issue_live_20261015.md"

with open(REPORT, "r", encoding="utf-8") as f:
    text = f.read()

# 查找所有图片引用
pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def fix_link(match):
    label = match.group(1)
    href = match.group(2)

    # 只处理图片
    if not href.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
        return match.group(0)

    # 如果已经是 ../figures/ 开头，已经正确
    if href.startswith("../figures/"):
        return match.group(0)

    # 报告在 report/shape_trials/gs_refine/ 目录下
    # 图片在 report/shape_trials/figures/ 目录下
    # 所以相对路径应该是 ../figures/filename.png
    new_rel = "../figures/" + href

    print(f"修复: {href} -> {new_rel}")
    return match.group(0).replace(href, new_rel)


new_text = pattern.sub(fix_link, text)

with open(REPORT, "w", encoding="utf-8") as f:
    f.write(new_text)

print("完成!")
