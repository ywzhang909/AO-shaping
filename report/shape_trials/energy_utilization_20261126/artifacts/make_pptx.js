/*
 * 实验汇报 PPT：整形对衍射效率的利用 (2026-11-26)
 * 复用本报告 figures/ 下已验证的 5 张图 + 7 张实拍证据图 + 17 张双周工作图 + 原生文本/表格。
 * 运行： NODE_PATH=<skill>/node_modules node make_pptx.js
 */
const pptxgen = require("pptxgenjs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const FIG = path.join(ROOT, "figures");
const OUT = path.join(ROOT, "energy_utilization_20261126.pptx");

const NAVY = "1F3A5F";
const TEAL = "2A9D8F";
const ORANGE = "E76F51";
const RED = "C0392B";
const GRAY = "7F8C8D";
const LIGHT = "F4F6F8";
const WHITE = "FFFFFF";
const FONT = "Microsoft YaHei";

// 实测图采集标注简写（完整三要素见附录 P29）
const CAP = {
  probe: "采集 11-23 12:53；0.02 ms / 3 帧 / 灰度 993；探针（详见 P29）",
  match: "采集 11-23 13:07–13:08；0.04 ms / 16 帧 / 暗场库；效率匹配（详见 P29）",
  abba: "采集 11-26 19:59:30–33；0.12 ms / 12 位 / A→B→B→A；ABBA（详见 P29）",
  pib: "采集 10-23；20 ms / 边长 100–1800 px；方斑扫描（详见 P29）",
};

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pptx.author = "AO-Shaping";
pptx.subject = "整形对衍射效率的利用：历史口径复算与闭环实测判定";
pptx.title = "整形对衍射效率的利用 (2026-11-26)";

// ---------- helper ----------
function header(slide, kicker, title) {
  slide.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: 13.333, h: 0.12, fill: { color: TEAL } });
  slide.addText(kicker, { x: 0.6, y: 0.3, w: 12.1, h: 0.28, fontFace: FONT, fontSize: 11, color: TEAL, bold: true, charSpacing: 1 });
  slide.addText(title, { x: 0.6, y: 0.58, w: 12.1, h: 0.6, fontFace: FONT, fontSize: 25, color: NAVY, bold: true });
  slide.addText("AO-Shaping · 2026-11-26", { x: 0.6, y: 7.02, w: 7, h: 0.3, fontFace: FONT, fontSize: 9, color: GRAY });
}

function footerNum(slide, n) {
  slide.addText(String(n), { x: 12.3, y: 7.0, w: 0.5, h: 0.3, fontFace: FONT, fontSize: 9, color: GRAY, align: "right" });
}

function callout(slide, text, x, y, w, h, color) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h,
    fill: { color: LIGHT },
    line: { color: color, width: 1.5 },
    rectRadius: 0.08,
  });
  slide.addText(text, { x: x + 0.16, y: y + 0.06, w: w - 0.32, h: h - 0.12, fontFace: FONT, fontSize: 12.5, color: NAVY, valign: "mid" });
}

function bullets(slide, items, x, y, w, h, fontSize) {
  const rows = items.map((it, i) => ({
    text: it,
    options: { bullet: true, breakLine: i !== items.length - 1, paraSpaceAfter: 8, fontSize: fontSize || 14, color: NAVY },
  }));
  slide.addText(rows, { x, y, w, h, fontFace: FONT, valign: "top" });
}

function img(slide, name, x, y, w, h) {
  slide.addImage({ path: path.join(FIG, name), x, y, w, h });
}

function captureNote(slide, text, x, y, w) {
  slide.addText(text, { x, y, w, h: 0.24, fontFace: FONT, fontSize: 8, color: GRAY, valign: "mid" });
}

// 等比缩放的缩略图网格：item = { file, ar, label }，ar = 原图宽/高。
function imgGrid(slide, items, x, y, cols, cellW, cellH, capH, fontSize) {
  items.forEach((it, i) => {
    const col = i % cols;
    const row = Math.floor(i / cols);
    const cx = x + col * cellW;
    const cy = y + row * (cellH + capH + 0.16);
    const boxW = cellW - 0.18;
    let w = boxW;
    let h = w / it.ar;
    if (h > cellH) {
      h = cellH;
      w = h * it.ar;
    }
    slide.addImage({ path: path.join(FIG, it.file), x: cx + (boxW - w) / 2, y: cy + (cellH - h) / 2, w, h });
    slide.addText(it.label, { x: cx, y: cy + cellH + 0.02, w: boxW, h: capH, fontFace: FONT, fontSize: fontSize || 9, color: GRAY, align: "center", valign: "top" });
  });
}

// ================= 1. 封面 =================
{
  const s = pptx.addSlide();
  s.background = { color: NAVY };
  s.addShape(pptx.ShapeType.rect, { x: 0, y: 2.35, w: 13.333, h: 0.06, fill: { color: TEAL } });
  s.addText("实验汇报", { x: 1.0, y: 1.5, w: 11.3, h: 0.5, fontFace: FONT, fontSize: 20, color: TEAL, bold: true, charSpacing: 3 });
  s.addText("整形对衍射效率的利用", { x: 1.0, y: 2.7, w: 11.3, h: 1.0, fontFace: FONT, fontSize: 42, color: WHITE, bold: true });
  s.addText("历史口径复算 · 倾斜与整形利用率 · A/B–ABBA 闭环实测判定", { x: 1.0, y: 3.75, w: 11.3, h: 0.5, fontFace: FONT, fontSize: 17, color: "C9D6E5" });
  s.addText("2026-11-26  |  AO-Shaping  |  只读归档数据，不重新采集", { x: 1.0, y: 6.3, w: 11.3, h: 0.4, fontFace: FONT, fontSize: 12, color: "9FB3C8" });
}

// ================= 2. 议程 =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "AGENDA", "汇报议程");
  const topics = [
    ["01", "核心结论速览", "一页看懂三组数据 + 一个判定"],
    ["02", "指标口径", "η = E_blaze / F0 与三种分母边界"],
    ["03", "倾斜探针效率", "合格口径 19.27% vs 旧口径虚高"],
    ["04", "F0 能量分解", "可转移 / 零级残余 / 未计入"],
    ["05", "同次效率匹配", "倾斜 54.21% vs 整形 −12.46%（趋势）"],
    ["06", "方形整形 PIB", "整帧口径，随边长增长"],
    ["07", "A/B–ABBA 判定", "变体定义 + paired_decision(cv) → refit_forward"],
    ["08", "结论与复现", "含 7 张实拍证据（跨级次剖面 / row·col 双向 / 三口径窗口）；数据真源与命令"],
    ["09", "双周工作总览", "11-16～11-26 共 22 个主题；图 W1–W17 为原报告归档图副本"],
  ];
  topics.forEach((t, i) => {
    const col = i < 5 ? 0 : 1;
    const row = i % 5;
    const x = 0.7 + col * 6.2;
    const y = 1.48 + row * 1.06;
    s.addText(t[0], { x, y, w: 0.8, h: 0.42, fontFace: FONT, fontSize: 19, color: TEAL, bold: true });
    s.addText(t[1], { x: x + 0.85, y, w: 5.1, h: 0.36, fontFace: FONT, fontSize: 14, color: NAVY, bold: true });
    s.addText(t[2], { x: x + 0.85, y: y + 0.36, w: 5.1, h: 0.5, fontFace: FONT, fontSize: 10, color: GRAY });
  });
  footerNum(s, 2);
}

// ================= 3. 核心结论速览 =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "EXECUTIVE SUMMARY", "核心结论速览");
  const cards = [
    { k: "19.27%", t: "倾斜探针合格口径 η", d: "旧口径 39.78% 虚高约 2 倍", c: NAVY },
    { k: "−12.46%", t: "整形同次效率净值 η_s", d: "不可判定，仅趋势参考", c: ORANGE },
    { k: "0.12 ms", t: "ABBA 非饱和同曝光", d: "四帧 peak 2019–2138", c: TEAL },
    { k: "接受", t: "paired_decision(cv)", d: "改善 0.0623 > 漂移 0.0360", c: TEAL },
  ];
  cards.forEach((c, i) => {
    const x = 0.7 + i * 3.15;
    s.addShape(pptx.ShapeType.roundRect, { x, y: 1.6, w: 2.95, h: 2.35, fill: { color: LIGHT }, line: { color: c.c, width: 1.5 }, rectRadius: 0.1 });
    s.addText(c.k, { x: x + 0.15, y: 1.85, w: 2.65, h: 0.75, fontFace: FONT, fontSize: 27, color: c.c, bold: true, align: "center" });
    s.addText(c.t, { x: x + 0.12, y: 2.68, w: 2.71, h: 0.55, fontFace: FONT, fontSize: 12.5, color: NAVY, bold: true, align: "center" });
    s.addText(c.d, { x: x + 0.12, y: 3.28, w: 2.71, h: 0.5, fontFace: FONT, fontSize: 10.5, color: GRAY, align: "center" });
  });
  callout(s, "一句话结论：口径先于数值——固定分母后谈效率；闭环均匀度用 ABBA 配对独立判定，采用 refit_forward。", 0.7, 4.35, 11.95, 1.0, TEAL);
  bullets(s, [
    "历史趋势只作参考：效率匹配不可判定，不构成谁优谁劣的判定。",
    "PIB 是整帧口径的集中度，与 η 分母不同，二者不可相除或互换。",
  ], 0.85, 5.7, 11.6, 1.1, 12.5);
  footerNum(s, 3);
}

// ================= 4. 指标口径 =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "DEFINITIONS", "指标定义与口径边界");
  s.addShape(pptx.ShapeType.roundRect, { x: 0.7, y: 1.45, w: 11.95, h: 1.05, fill: { color: NAVY }, line: { color: NAVY }, rectRadius: 0.08 });
  s.addText("η = E_blaze / F0", { x: 0.95, y: 1.6, w: 5.2, h: 0.7, fontFace: FONT, fontSize: 28, color: WHITE, bold: true, valign: "mid" });
  s.addText("F0 = 平相位帧零级窗口暗减积分\nE_blaze = 闪耀帧闪耀级窗口暗减积分", { x: 6.2, y: 1.55, w: 6.2, h: 0.9, fontFace: FONT, fontSize: 12, color: "C9D6E5", valign: "mid" });
  const rows = [
    [{ text: "指标", options: { bold: true, color: WHITE, fill: { color: NAVY } } }, { text: "分子", options: { bold: true, color: WHITE, fill: { color: NAVY } } }, { text: "分母", options: { bold: true, color: WHITE, fill: { color: NAVY } } }, { text: "用途", options: { bold: true, color: WHITE, fill: { color: NAVY } } }],
    ["η（主口径）", "闪耀级窗口 E_blaze", "平相位零级窗口 F0", "可转移能量份额"],
    ["旧口径闪耀级份额", "闪耀级窗口能量", "闪耀帧三衍射级窗口之和 S3", "帧内分配，诊断用"],
    ["PIB", "目标方框内能量", "CCD 整帧能量", "整形集中度"],
  ];
  s.addTable(rows, { x: 0.7, y: 2.85, w: 11.95, colW: [2.6, 3.3, 3.1, 2.95], rowH: 0.62, fontFace: FONT, fontSize: 12.5, color: NAVY, border: { type: "solid", color: "D5DEE8", pt: 1 }, fill: { color: WHITE }, valign: "mid" });
  callout(s, "不使用 lobe.ratios[blaze_order]：其分母是闪耀帧三衍射级窗口之和，同一探针读数 39.78% vs 19.27%，虚高约 2 倍。", 0.7, 5.35, 11.95, 0.85, ORANGE);
  footerNum(s, 4);
}

// ================= 5. 倾斜探针效率（图1） =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "TILT / BLAZE", "倾斜（闪耀光栅）：探针口径效率");
  img(s, "fig1_eta_compare.png", 0.95, 1.35, 8.4, 4.58);
  const x = 9.6;
  s.addText("关键点", { x, y: 1.45, w: 3.0, h: 0.4, fontFace: FONT, fontSize: 15, color: TEAL, bold: true });
  bullets(s, [
    "横轴是探测参数（周期 16 px / x 轴 / 灰度深度 993 / 有效曝光），不是日期。",
    "合格配置 P16 · 0.02 ms(eff)：请求 0.1 ms 经自动曝光稳定到 0.02 ms，η = 19.27%（minus_one），级次间距 471 px。",
    "P16 · 5.0 ms 饱和（峰值 255）、P16 · 0.1 ms 的 ±1 级能量为 0，均无判据资格。",
  ], x, 1.95, 3.05, 3.8, 11.5);
  captureNote(s, CAP.probe, x, 5.78, 3.05);
  callout(s, "主口径 η = E_blaze / F0 = 19.27%；旧口径闪耀级份额 = E_blaze / S3 = 39.78%，虚高来自分母而非效率提升。", 0.95, 6.08, 11.7, 0.78, TEAL);
  footerNum(s, 5);
}

// ================= 6. F0 能量分解（图2） =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "ENERGY BUDGET", "合格探针的 F0 能量去向");
  img(s, "fig2_f0_breakdown.png", 1.1, 1.6, 11.1, 3.5);
  const stats = [
    ["19.27%", "闪耀级可转移 η", TEAL],
    ["29.18%", "零级残余 / F0", NAVY],
    ["51.54%", "三衍射级窗口未计入 / F0", GRAY],
  ];
  stats.forEach((st, i) => {
    const x = 1.1 + i * 3.75;
    s.addText(st[0], { x, y: 5.18, w: 3.5, h: 0.5, fontFace: FONT, fontSize: 24, color: st[2], bold: true, align: "center" });
    s.addText(st[1], { x, y: 5.7, w: 3.5, h: 0.35, fontFace: FONT, fontSize: 12, color: NAVY, align: "center" });
  });
  captureNote(s, CAP.probe, 1.1, 6.12, 11.1);
  callout(s, "窗口截断、无几何标定，「可转移份额」≠「帧内占比」，两者是完全不同的问题。", 1.1, 6.5, 11.1, 0.55, ORANGE);
  footerNum(s, 6);
}

// ================= 7. 同次效率匹配（图3） =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "SAME-SHOT MATCH", "同次效率匹配：倾斜 vs 整形");
  img(s, "fig3_efficiency_match.png", 0.95, 1.35, 8.1, 4.68);
  const x = 9.3;
  s.addText("判定：不可判定（undeterminable）", { x, y: 1.45, w: 3.4, h: 0.4, fontFace: FONT, fontSize: 15, color: RED, bold: true });
  bullets(s, [
    "同一 F0 圆盘口径：倾斜 η_d = 54.21%，整形 η_s = −12.46%。",
    "d 均值 −0.6667，95% CI [−0.7068, −0.6388]。",
    "三条判据未过：能量检查越界、G1 组内 SD 超限、漂移轮次 > 1。",
  ], x, 1.95, 3.4, 3.8, 12);
  captureNote(s, CAP.match, x, 5.78, 3.4);
  callout(s, "只作趋势参考，不构成谁优谁劣的判定。", 0.95, 6.25, 7.9, 0.6, ORANGE);
  footerNum(s, 7);
}

// ================= 8. 实拍证据：三态远场（图6） =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "RAW CAPTURES / 1", "实拍证据：平相位 / 倾斜 / 整形三态远场");
  img(s, "fig6_real_efficiency_match.png", 0.7, 1.45, 11.95, 3.85);
  bullets(s, [
    "来源：efficiency_match/20261123_050735Z/raw 的 0000/0001/0002 帧，暗减后统一 1–2100 counts 对数色标。",
    "倾斜帧可见 0 级与 +1 级双斑，整形帧能量摊开——与图 3 的 η_d / η_s 趋势同向。",
  ], 0.85, 5.35, 11.6, 0.95, 12);
  captureNote(s, CAP.match, 0.85, 6.32, 11.6);
  callout(s, "只作可视核对：该次 verdict=undeterminable，不能据图判定谁优谁劣。", 0.85, 6.6, 11.6, 0.42, ORANGE);
  footerNum(s, 8);
}

// ================= 9. 实拍证据：跨 0 级 / ±1 级剖面（图9） =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "RAW CAPTURES / 2", "实拍证据：跨 0 级 / ±1 级的亮度剖面");
  img(s, "fig9_real_lobe_profile.png", 0.85, 1.4, 11.6, 4.5);
  bullets(s, [
    "沿图 6 青色竖线 col=1230，col=1226–1234 九列平均；三帧同次 0.04 ms 曝光、未归一化。",
    "倾斜帧 0 级约 0.85k、+1 级（row=1047）约 0.52k 独立峰；−1 级仅弱残余。级次坐标取自 analysis.json；剖面峰高不是窗口积分，η 仍以图 3 口径为准。",
  ], 0.85, 5.98, 11.6, 0.75, 11);
  captureNote(s, CAP.match, 0.85, 6.76, 11.6);
  footerNum(s, 9);
}

// ================= 10. 实拍证据：横向 col 剖面（图10） =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "RAW CAPTURES / 3", "实拍证据：横向 col 剖面——圆盘口径 vs 方框口径");
  img(s, "fig10_real_col_profile.png", 0.85, 1.4, 11.6, 4.5);
  bullets(s, [
    "与图 9 正交：row=1102–1108 七行平均，col=1080–1380；同次 0.04 ms 曝光、暗减后、未归一化。0 级中心 col≈1227.18 取自 analysis.json。",
    "淡蓝带为 η 分母 F0 圆盘（r0=48 px）横向区间，淡绿带为目标方框（side=61.2 px，PIB / η_s 口径）；横向无分开的 ±1 级峰，级次主轴沿 row。",
  ], 0.85, 5.98, 11.6, 0.75, 11);
  captureNote(s, CAP.match, 0.85, 6.76, 11.6);
  footerNum(s, 10);
}

// ================= 11. 实拍证据：三口径窗口在同一剖面（图11） =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "RAW CAPTURES / 4", "实拍证据：三种口径窗口落在同一剖面上");
  img(s, "fig11_real_metric_windows.png", 0.55, 1.4, 12.2, 3.66);
  bullets(s, [
    "左：平相位 F0 圆盘（r0=48）为 η 分母；中：倾斜帧 G1(+1)/GM1(−1) 圆盘（r1=60）与 G0 圆盘，η 分子取 G1；右：整形帧目标方框（side=61.2）为 η_s 分子 / PIB 口径。",
    "竖线是二维窗口在剖面上的穿越边界，不是积分结果；窗口参数取 analysis.json thresholds 与各帧 reference_center。图 3 新旧口径差约 2 倍，根源在分母窗口不同。",
  ], 0.85, 5.2, 11.6, 1.45, 11);
  captureNote(s, CAP.match, 0.85, 6.7, 11.6);
  footerNum(s, 11);
}

// ================= 12. 方形整形 PIB（图4） =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "SQUARE SHAPING", "方形整形 PIB 随方框尺寸增长：与 0 级能量的量级对照");
  img(s, "fig4_pib.png", 0.85, 1.35, 11.6, 4.3);
  bullets(s, [
    "PIB = 方框内能量 / CCD 整帧能量。20 ms 组 100→1800 px：43.53%→96.88%（≥200 px 未饱和）。",
    "180 ms 组全饱和（红圈），只作趋势，最大 1700 px 为 95.75%；PIB 与 η 分母不同，不可相除。",
  ], 0.95, 5.72, 5.8, 1.0, 11);
  bullets(s, [
    "0 级参照：器件地板 3.03%（P=32，三衍射级窗口归一）、ramp 993 为 4.16%；探针零级残余 29.18%（F0 分母）。",
    "三种分母与条件不同，仅量级对照；方形扫描 CSV 无逐尺寸 0 级能量列，不做逐点配对。",
  ], 6.85, 5.72, 5.6, 1.0, 11);
  captureNote(s, CAP.pib, 0.95, 6.75, 11.5);
  footerNum(s, 12);
}

// ================= 13. A/B–ABBA 判定（图5） =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "A/B–ABBA DECISION", "闭环实测判定：高清冻结基线 (A) vs 重拟合候选 (B)");
  img(s, "fig5_abba_cv.png", 0.85, 1.35, 7.5, 4.27);
  const rows = [
    [{ text: "量", options: { bold: true, color: WHITE, fill: { color: NAVY } } }, { text: "值", options: { bold: true, color: WHITE, fill: { color: NAVY } } }],
    ["高清冻结基线 A（均值 CV）", "1.915007"],
    ["重拟合候选 B（均值 CV）", "1.852745"],
    ["drift", "0.035982"],
    ["tolerance", "0.0"],
    ["改善", "0.062261"],
    [{ text: "接受（accepted=True）", options: { bold: true, color: TEAL } }, { text: "采用 refit_forward", options: { bold: true, color: TEAL } }],
  ];
  s.addTable(rows, { x: 8.65, y: 1.5, w: 4.0, colW: [1.9, 2.1], rowH: 0.5, fontFace: FONT, fontSize: 12, color: NAVY, border: { type: "solid", color: "D5DEE8", pt: 1 }, fill: { color: WHITE }, valign: "mid" });
  callout(s, "A=高清冻结基线：640×640/FFT 2048、逐位继承、不重拟合；B=重拟合候选：从 A 热启动，train-only 重拟合物理参数，架构与推理口径不变。", 0.85, 5.06, 11.8, 0.64, GRAY);
  callout(s, "条件：0.12 ms 非饱和、同曝光、单变量配对；ABBA 帧序 0_A/1_B/2_B/3_A 对应基线 A→候选 B→候选 B→基线 A。判据：改善 0.062261 > 漂移 0.035982 + 容差 0。", 0.85, 5.78, 11.8, 0.82, TEAL);
  captureNote(s, CAP.abba, 0.85, 6.68, 11.8);
  footerNum(s, 13);
}

// ================= 14. 变体定义：两个模型分别是什么 =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "VARIANT DEFINITIONS", "直观命名：高清冻结基线 (A) 与 重拟合候选 (B)");
  const rows = [
    [{ text: "项目", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
     { text: "高清冻结基线（model640 / A）", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
     { text: "重拟合候选（refit_forward / B）", options: { bold: true, color: WHITE, fill: { color: NAVY } } }],
    ["角色", "冻结基线前向模型（未重拟合）", "同一物理前向在数据上重拟合后的候选"],
    ["来源", "原始拟合检查点经 geometric_seed 升控制分辨率，光学/辐射参数逐位继承", "以 model640 为 base_checkpoint，用 scripts/slm/refit_transfer_forward.py 重拟合"],
    ["关键配置", "grid 640 / FFT 2048 / bias_counts=0.0；background 5.89、gain 668.7", "radiometry=exposure · adaptation=full · 100 iters · patience 8 · min_delta 1e-4"],
    ["数据使用", "原始捕获集拟合（校准集 9 / 训练集 3 / 验证集 2 / 测试集 0）", "训练集 3 拟合、验证集 2 选优；校准集 9 排除；测试集 0"],
    ["验证集表现", "rel_l2 0.5859 · corr 0.8574", "rel_l2 0.5747 · corr 0.8744（两指标均改善）"],
    ["相位真源", "ab_validate_base/square_best.npy · sha256 2907ef3b…", "ab_validate_refit/square_best.npy · sha256 946e6196…"],
  ];
  s.addTable(rows, {
    x: 0.7, y: 1.55, w: 11.95, colW: [2.1, 4.85, 5.0], rowH: 0.62,
    fontFace: FONT, fontSize: 11, color: NAVY,
    border: { type: "solid", color: "D5DEE8", pt: 1 }, fill: { color: WHITE }, valign: "mid",
  });
  callout(s, "两者只有「前向模型/相位来源」不同：同一 SLM #1、同一冻结 core+camera mask、同一 0.12 ms 非饱和曝光、同一方形目标 → ABBA 单变量配对成立。", 0.7, 6.2, 11.95, 0.7, TEAL);
  footerNum(s, 14);
}

// ================= 15. 模型结构演化：原始模型 → A → B =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "MODEL LINEAGE / 结构对照", "模型结构演化：原始模型 → 高清冻结基线 → 重拟合候选");
  s.addImage({ path: path.join(FIG, "fig_variant_structure.png"), x: 0.75, y: 1.28, w: 11.84, h: 5.18 });
  callout(s, "同一套可微物理预测，三个检查点：A 只升级控制分辨率且不重拟合；B 在 A 上热启动、仅训练集重拟合物理参数，架构与推理口径不变。", 0.75, 6.55, 11.84, 0.65, TEAL);
  footerNum(s, 15);
}

// ================= 16. 实拍证据：ABBA 四帧（图7） =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "RAW CAPTURES / 5", "实拍证据：A/B–ABBA 四帧远场");
  img(s, "fig7_real_abba_frames.png", 0.7, 1.5, 11.95, 3.3);
  bullets(s, [
    "来源：ab_replay_square_retry2/captures 的 0_A.npz（基线 A 首帧）、1_B.npz（候选 B 第 1 帧）、2_B.npz（候选 B 第 2 帧）、3_A.npz（基线 A 末帧）；四帧统一 1–2200 counts 对数色标。",
    "青色虚线是 abba_core_mask_camera.npz 的 CV 统计窗口；变体 B 两帧旁瓣更弱，方向与图 5 一致。",
  ], 0.85, 5.0, 11.6, 1.05, 12);
  callout(s, "实拍只核对方向；判定仍严格以 paired_decision(metric=\"cv\") 的「接受（accepted=True）」为准。", 0.85, 6.2, 11.6, 0.45, TEAL);
  captureNote(s, CAP.abba, 0.85, 6.68, 11.6);
  footerNum(s, 16);
}

// ================= 17. 实拍证据：ABBA 四帧双向剖面（图12） =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "RAW CAPTURES / 6", "实拍证据：ABBA 四帧 row / col 双向亮度剖面");
  img(s, "fig12_real_abba_profiles.png", 0.55, 1.4, 12.2, 3.94);
  bullets(s, [
    "row（col=1224–1231 七列平均）与 col（row=1101–1108 七行平均）正交方向；四帧未暗减、同次 0.12 ms 曝光、未归一化。",
    "青色带与灰虚线是 CV 掩膜外接框（row 1068–1140、col 1191–1263）；B 帧中心略低、能量更分散，方向与 CV 改善一致，判定仍以 paired_decision(metric=\"cv\") 为准。",
  ], 0.85, 5.48, 11.6, 1.25, 11);
  captureNote(s, CAP.abba, 0.85, 6.76, 11.6);
  footerNum(s, 17);
}

// ================= 18. 实拍证据：方斑远场（图8） =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "RAW CAPTURES / 7", "实拍证据：方斑边长扫描远场");
  img(s, "fig8_real_square_shaping.png", 0.9, 1.45, 6.9, 4.97);
  const x = 8.1;
  s.addText("与图 4 对应", { x, y: 1.5, w: 4.3, h: 0.4, fontFace: FONT, fontSize: 15, color: TEAL, bold: true });
  bullets(s, [
    "选帧依据：hw_gs_square_scans/.../frames_selection.json。",
    "边长 200/600/1000/1500/1800 px 对应 PIB 0.548/0.691/0.802/0.924/0.969。",
    "原始 npz 未随归档保留，图由该批次实拍远场重绘。",
    "PIB 是整帧口径，不能与 η 互换。",
  ], x, 2.05, 4.4, 3.9, 12);
  captureNote(s, CAP.pib, x, 6.0, 4.4);
  callout(s, "实拍形态与图 4 的单调增长趋势一致。", x, 6.3, 4.4, 0.5, ORANGE);
  footerNum(s, 18);
}

// ================= 19. 双周工作总览：上周 =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "BIWEEKLY / 1", "双周工作总览：上周 11-16—11-22");
  bullets(s, [
    "M3/保真度：方向一致性达到判据 ≠ 强度保真度达到判据；「7.5× 分叉」撤回；corr +0.9497，基座 15.0 counts vs 2.8847。",
    "#247 轴映射：安装旋转 +88.544°，对齐后跨轴 ≤0.12 px；A1b RMS 0.871 px，frozen=false。",
    "速度解耦：30/30 ok，实跑 2.284 fps（计划 10）；SLM 写入+换槽 0.312 s vs CCD 0.075 s。",
    "#238：2π=993；二值 g₂π≈964±13；峰值法 938±11 撤回，缺同曝光暗场不发布。",
    "P4 双门有效 fail：relL2 0.88489 > δ=0.20；方向 4/14 < 10/14；三处泄漏先修。",
    "工程：286 commit（95 个 #205）、CLI 21 命令、#253 σ_total 0.0626–0.8952 rad。",
  ], 0.7, 1.5, 5.7, 4.6, 11);
  imgGrid(s, [
    { file: "bw_w01_speed_fps.png", ar: 2.2222, label: "图 W1：30 条 transition 实测 fps vs 计划 10 fps" },
    { file: "bw_w02_speed_timing.png", ar: 1.5556, label: "图 W2：CCD 0.075 s vs SLM 写入+换槽 0.312 s" },
  ], 6.6, 1.5, 1, 5.9, 1.85, 0.32, 9);
  callout(s, "上周主线：模型前向假设被连续证伪；速度瓶颈定位在 SLM 侧，不在相机侧。", 0.7, 6.3, 11.9, 0.6, TEAL);
  footerNum(s, 19);
}

// ================= 20. 双周工作总览：上周证据图 =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "BIWEEKLY / 2", "双周工作总览：上周光路与参数证据");
  bullets(s, [
    "理想光路静态测试：5 图案 peak 95/90/94/90/128，全部 observed=false；方形平顶未出现。",
    "透镜三联：相位→下发灰度→CCD 远场；模型预期方形平顶，实测点状聚焦。",
    "--debug-weights：训练损失 0.39325→0.34548，6 轮全部 accepted=false。",
    "在线参数：0→1 轮跳变一次后持平；拟合迭代与闭环轮次不能对齐。",
  ], 0.7, 1.5, 5.7, 3.8, 12);
  imgGrid(s, [
    { file: "bw_w03_ideal_overview.png", ar: 4.6875, label: "图 W3：5 图案远场总览（0.05 ms）" },
    { file: "bw_w04_ideal_lens.png", ar: 3.0556, label: "图 W4：透镜相位/灰度/远场三联" },
    { file: "bw_w05_online_round.png", ar: 1.3333, label: "图 W5：在线参数按闭环轮次 0–5" },
    { file: "bw_w06_online_iteration.png", ar: 1.3333, label: "图 W6：训练损失按拟合迭代 0–6" },
  ], 6.6, 1.45, 2, 3.05, 1.72, 0.32, 8.5);
  callout(s, "没有归档图的主题（#235 / #238 / P4 / 工程治理）已在报告 §9.1 表中列出，不补画示意图。", 0.7, 6.3, 11.9, 0.6, ORANGE);
  footerNum(s, 20);
}

// ================= 21. 双周工作总览：本周 #247 =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "BIWEEKLY / 3", "双周工作总览：本周口径与 #247 归因");
  bullets(s, [
    "η=E_blaze/F0：19.27% vs 旧 39.78%；ABBA 接受（accepted=True），采用 refit_forward。",
    "S1e A：H 两两最大 2.431 px > 1.5；H/B=1.6004≈1920/1200；位置抖动非轴翻转。",
    "S1e P：H 1.585 px，跨变体 D5 1.389 px < 1.5；物理断电无效应，E2 维持。",
    "波形族：92 帧，x 族 89.84° pass / y 族 1.18° fail，验证集 0/10 ⇒ not_assessable。",
    "根因：F-contiguous 数组被裸 memmove 转置；守卫后 SHA b98be9a6…→7276d9da…，42 项 pytest 全过。",
    "重复性：276 帧，reproduced 15 / not_reproduced 0 / not_assessable 5。",
  ], 0.7, 1.5, 5.7, 4.6, 11);
  imgGrid(s, [
    { file: "bw_w07_s1e_arm_a.png", ar: 3.2609, label: "图 W7：S1e 变体 A 三联图" },
    { file: "bw_w08_s1e_arm_p.png", ar: 3.2609, label: "图 W8：S1e 变体 P（断电重启）三联图" },
  ], 6.6, 1.6, 1, 5.9, 1.8, 0.32, 9);
  callout(s, "判据采前冻结、采后不改；S1 维持 BLOCKED，重采待授权。", 0.7, 6.3, 11.9, 0.6, RED);
  footerNum(s, 21);
}

// ================= 22. 双周工作总览：本周 transfer 双门 =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "BIWEEKLY / 4", "双周工作总览：本周迁移双门分离");
  bullets(s, [
    "点扩散函数验证集：0.60243929→0.43018529（−28.5928%），CI (0.087282, 0.276039)。",
    "点扩散函数独立 192 px：0.52978986→0.41696567（−21.2960%），CI (0.040656, 0.193446)。",
    "几何验证集：0.6200% < 5%；独立 y 轴 +18.962% > 10% ⇒ learn_geometry=False。",
    "结论：静态点扩散函数作为显式启用项，默认不启用点扩散函数学习；几何残差保持默认关闭。",
  ], 0.7, 1.5, 5.7, 3.8, 11);
  imgGrid(s, [
    { file: "bw_w09_psf_validation.png", ar: 2.0463, label: "图 W9：点扩散函数验证集相对 L2" },
    { file: "bw_w10_psf_independent.png", ar: 2.1284, label: "图 W10：点扩散函数独立 192 px" },
    { file: "bw_w11_geometry_validation.png", ar: 2.0463, label: "图 W11：几何验证集未过 5% 门" },
    { file: "bw_w12_geometry_holdout.png", ar: 1.8800, label: "图 W12：几何验证集 y 轴 +18.962%" },
  ], 6.6, 1.45, 2, 3.05, 1.72, 0.32, 8.5);
  callout(s, "同一批数据上「点扩散函数口径达到预注册判据、几何口径未达到判据」：残差主要是各向同性展宽，不是几何偏差。", 0.7, 6.3, 11.9, 0.6, TEAL);
  footerNum(s, 22);
}

// ================= 23. 双周工作总览：本周 HMWS / 速度 / 可微 =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "BIWEEKLY / 5", "双周工作总览：本周 HMWS / 速度 / 可微");
  bullets(s, [
    "HMWS：Noll 5 可用（12-bit slope 0.8662），Noll 4 失效（8-bit 0.3947 / 12-bit 0.1772）；位深假设证伪。",
    "焦扫：status=ambiguous、accepted=false，未做焦面补偿。",
    "闭环速度：209.205 s + 间隙 0.390 s；采集 84.33 s（40.3%）；异步写盘约省 14 s 未接线。",
    "K 轴：CV 0.28381→0.28365（0.05%），墙钟 +17%～60%；L-BFGS 第 5 迭代 ≈0.67 s vs 5.91 s（≈8.9×）。",
  ], 0.7, 1.45, 5.7, 4.3, 10.5);
  imgGrid(s, [
    { file: "bw_w13_hmws_linearity.png", ar: 1.4545, label: "图 W13：Zernike 注入/恢复线性" },
    { file: "bw_w14_hmws_focus.png", ar: 1.6667, label: "图 W14：焦面扫描 ambiguous" },
    { file: "bw_w15_tsne_anchored.png", ar: 1.2308, label: "图 W15：实测校正相位 t-SNE" },
    { file: "bw_w16_tsne_fit.png", ar: 1.2308, label: "图 W16：Zernike 拟合轨迹 t-SNE" },
    { file: "bw_w17_kaxis.png", ar: 1.7143, label: "图 W17：K 轴 CV 随迭代" },
  ], 6.6, 1.4, 2, 3.05, 1.34, 0.28, 8);
  callout(s, "两周收束见报告 §9.3：口径先固定、达到判据才启用、速度瓶颈在 SLM 写入+换槽。", 0.7, 6.3, 11.9, 0.6, TEAL);
  footerNum(s, 23);
}

// ================= 24. 结论 =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "CONCLUSIONS", "结论与最优路线");
  bullets(s, [
    "口径先于数值：同一实验旧口径 39.78% vs 新口径 19.27%；主口径 η = E_blaze / F0 衡量从平相位 0 级可转移的份额。",
    "倾斜更「能拿光」是趋势不是判定：探针 19.27%、效率匹配 54.21% 均高于整形净值 −12.46%，但后者不可判定。",
    "闭环方案判定独立成立：ABBA 配对接受（accepted=True），采用 refit_forward；改的是均匀度，不改写利用率趋势。",
    "PIB 不可当效率：整帧口径的集中度指标，与 η 分母不同。",
    "最优路线：先用非饱和同曝光配对锁定闭环方案，再谈利用率数值；不同分母分开报告、互不代入；实拍证据见图 6–12。",
    "剖面只能从归档原始帧导出：效率匹配三态与 ABBA 四帧已补 row/col 剖面；合格探针与 PIB 无原始帧，不补（见报告 §6.4）。",
    "双周工作总览见报告 §9：上周 10 主题 + 本周 12 主题，图 W1–W17 为原报告归档图副本，不新增判定。",
  ], 0.85, 1.55, 11.7, 4.7, 12);
  callout(s, "本周重点方向：闭环整形均匀度与能量利用率口径的统一与可复现判定；动态整形方向沿用本报告确立的配对框架。", 0.85, 6.25, 11.7, 0.7, TEAL);
  footerNum(s, 24);
}

// ================= 25. 时间分析（补充） =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "TIMING / 时间分析", "最新实测会话分阶段耗时：采集是最大头");
  img(s, "fig13_stage_timing.png", 0.55, 1.45, 7.3, 3.36);
  bullets(s, [
    "2026-11-24 实测会话总耗时 ≈209 s（journal.json，host 调用口径，含 24 样本采集 + 拟合 + 4×5 反馈优化块）。",
    "采集 84.3 s 占 40%：每样本 ≈3.5 s = 写相位 + 稳定 + 3 帧过渡 + 取帧，是第一改进对象。",
    "实测校正反馈优化 48.3 s：块内 5 个 Adam 步各 25–40 ms，块间采集/评估 ≈6.0 s——优化本身快，硬件事务是瓶颈。",
    "拟合 34.8 s、CCD 模型 38.0 s；audit + summary ≈3.8 s。",
  ], 8.05, 1.5, 4.75, 3.2, 11.5);
  callout(s, "改进点：丢帧 3→2、每样本帧数 3→2、稳定等待由叠加改为取 max（须先验证 settle 判据不提前通过）；闭环对外结论仍须交错重放支撑。", 0.55, 4.85, 12.3, 0.85, ORANGE);
  callout(s, "数据缺口：GUI 闭环（target-lock）2026-11-23 两次运行均第 0 轮 fault，无多轮逐项耗时归档；需在写入器补 per-stage 计时后重跑一轮。", 0.55, 5.85, 12.3, 0.7, GRAY);
  footerNum(s, 25);
}

// ================= 26. 梯度方向一致性（补充） =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "GRADIENT DIRECTION / 方向一致性", "梯度方向一致性：像素级与低维两套口径");
  s.addImage({ path: path.join(ROOT, "images", "gradient_direction_consistency_20261116.png"), x: 0.7, y: 1.5, w: 2.22, h: 4.17 });
  bullets(s, [
    "可微相位优化：逐轮相位可逐位回放（loss max_rel_diff=0）；active 像素 4.9%，相邻增量符号一致性均值 0.611，单调(≥0.9)像素仅 0.06%。",
    "lr 反事实：0.02/0.05/0.08/0.12 最终 loss 差约 2%/0%/8%/22%，仅 lr=0.05 第 98 轮达存档目标 ⇒ 方向不一致，不宜放大 lr。",
    "低维对照：迁移拟合阶段 Zernike 系数后段方向余弦中位 0.9978、91.2% 轨迹后段一致，与像素级动力学不同，lr 处方不可混用。",
    "负对照：孪生帧 sim 同批次方向余弦中位 0.0288、符号一致率 0.5056，门限 0.5 下无方向信息。",
  ], 3.15, 1.5, 9.65, 4.3, 11.5);
  callout(s, "结论：像素级相位方向不一致时，放大 lr 只放大噪声与失真；宜保持或减小 lr。图源：diff_phase_monotonicity_replay_20261121。", 0.55, 5.95, 12.3, 0.75, TEAL);
  footerNum(s, 26);
}

// ================= 27. 不同 lr 远场光斑演化 GIF（实测 CCD） =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "ANIMATION / lr 远场演化", "不同 lr 下远场光斑随迭代变化（实测 CCD GIF，2026-11-27）");
  s.addImage({ path: path.join(ROOT, "images", "lr_farfield_hw_ts_fullframe.gif"), x: 0.55, y: 1.55, w: 7.6, h: 2.73 });
  bullets(s, [
    "来源：data/debug/time_to_target_20261127/frames1；四档 lr（0.02/0.05/0.08/0.12）× 20 迭代，原始 CCD 帧未暗减。",
    "每帧叠加 journal.jsonl :: timestamp_utc 的 UTC+8 时刻；四档 lr 依次采集，同一迭代跨档时刻不同。",
    "2592 px 全画幅窗、全局归一化；用于展示实测光斑演化，不作为均匀度判定，判定仍以 §5 配对 CV 为准。",
  ], 8.4, 1.55, 4.4, 2.9, 11.5);
  callout(s, "说明：GIF 为实测 CCD 帧（非离线 |FFT|²），渲染脚本 scripts/slm/render_lr_farfield_hw_gif.py；时间戳与帧清单见 figures/lr_farfield_hw_ts_manifest.json。", 0.55, 4.6, 12.3, 0.85, GRAY);
  callout(s, "对比参照：lr_sweep_20261116 六档（0.01–0.4）只落标量指标；本页改用 time_to_target_20261127 的实测多 lr × 逐迭代序列。", 0.55, 5.6, 12.3, 0.7, ORANGE);
  footerNum(s, 27);
}

// ================= 28. 数据来源与复现 =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "PROVENANCE", "数据来源与可复现");
  const rows = [
    [{ text: "内容", options: { bold: true, color: WHITE, fill: { color: NAVY } } }, { text: "真源 / 复算", options: { bold: true, color: WHITE, fill: { color: NAVY } } }],
    ["历史三口径", "scripts/slm/analyze_diffraction_vs_shaping_utilization_20261126.py --json"],
    ["探针明细", "data/diffraction_efficiency/probe_real_20261123_gate/summary.json"],
    ["效率匹配", "data/debug/efficiency_match/20261123_050735Z/analysis.json"],
    ["ABBA 判定", ".../ab_replay_square_retry2/summary.json · paired_decision(cv)"],
    ["双周工作总览", "报告 §9 · 图 W1–W17 为原报告 figures/ 副本，映射见 §8.1"],
    ["变体定义", "ab_session_20261124/refit_forward/refit_provenance.json · model640/frozen_parameters.json · watch_20261124_1406/summary.json"],
    ["0 级能量参照", "hw_blaze_20261023/final_numbers.json（3.03% / 4.16%）· 合格探针 summary.json（29.18%）"],
    ["图表数据", "artifacts/data.json → make_figures.py"],
    ["实拍证据", "export_real_captures.py（跨级次剖面 / row·col 双向 / 三口径窗口）；hw_gs_square_scans/.../frames_selection.json"],
    ["剖面数据边界", "报告 §6.4：合格探针与 PIB 无原始帧，不补剖面"],
    ["上级报告", "gui_closed_loop_energy_efficiency_20261125.md"],
    ["时间分析（P25）", "data/debug/slm_transfer/20261124_200949/journal.json · artifacts/make_timing_figure.py"],
    ["方向一致性（P26）", "diff_phase_monotonicity_replay_20261121.md · data/debug/differentiable_opt/phase_monotonicity_20261116.png"],
    ["lr 远场 GIF（P27）", "data/debug/time_to_target_20261127/frames1 · journal.jsonl :: timestamp_utc · scripts/slm/render_lr_farfield_hw_gif.py"],
  ];
  s.addTable(rows, { x: 0.7, y: 1.5, w: 11.95, colW: [2.6, 9.35], rowH: 0.42, fontFace: FONT, fontSize: 11, color: NAVY, border: { type: "solid", color: "D5DEE8", pt: 1 }, fill: { color: WHITE }, valign: "mid" });
  callout(s, "溯源登记与页面改动按整文件让渡给持有车道，待落地后合入并重跑 sync_report_provenance.py --check / ruff check . / graphify update .", 0.7, 6.3, 11.95, 0.65, ORANGE);
  footerNum(s, 28);
}

// ================= 29. 实测图采集元数据（附录） =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "APPENDIX / CAPTURE METADATA", "实测图采集元数据：时间 / 参数 / 来源算法");
  const hd = (t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: NAVY } } });
  const rows = [
    [hd("图"), hd("采集时间（UTC+8）"), hd("设置参数"), hd("来源算法 / 脚本")],
    ["图 1、2", "2026-11-23 12:53:11", "SLM #1（1064 nm）、大恒相机 #0；请求曝光 0.1 ms / 生效 0.02 ms（自动曝光）；3 帧、稳定 0.25 s；灰度深度 993；周期 16 px、x 轴、间距 471 px、−1 级；暗减；面板 1200×1920",
      "连续闪耀光栅衍射效率探针，src/ao_shaping/tools/slm/calibration/diffraction_efficiency/measure.py；data/diffraction_efficiency/probe_real_20261123_gate/summary.json"],
    ["图 3、6、9、10、11", "2026-11-23 13:07:42–13:08:11（分析 13:19:36）", "曝光 0.04 ms；16 帧矩阵（平相位 6 / 倾斜 4 / 整形 4 / 暗场 2）；面板 1200×1920；满量程 4095；暗场库 dark_library_0.04ms.npy",
      "GS 方斑相位 + 连续闪耀光栅，scripts/slm/efficiency_match_capture_20261123.py、efficiency_match_analyze_20261123.py"],
    ["图 4、8", "2026-10-23（批次 20261023_gs_square）", "曝光 20 ms；边长 100–1800 px；原始帧 2592×1944",
      "GS 方斑整形扫描，scripts/slm/hw_gs_square_scans.py；frames_selection.json"],
    ["图 5、7、12", "2026-11-26 19:59:30–19:59:33（设备 19:59:27 开、19:59:35 释放）", "曝光 0.12 ms、12 位；稳定等待 120 s；序列 A→B→B→A；CV 掩膜 abba_core_mask_camera.npz；峰值 2059–2138，未饱和",
      "冻结物理预测模型基线 A（model640，相位 sha256 2907ef3b…）vs 重拟合物理预测模型 B（refit_forward，相位 sha256 946e6196…），scripts/slm/replay_closed_loop_abba_20261124.py"],
  ];
  s.addTable(rows, { x: 0.35, y: 1.45, w: 12.6, colW: [1.45, 2.5, 4.5, 4.15], rowH: 1.02, fontFace: FONT, fontSize: 9, color: NAVY, border: { type: "solid", color: "D5DEE8", pt: 1 }, fill: { color: WHITE }, valign: "mid" });
  callout(s, "图 W1–W17 为跨报告归档副本，其采集时间、设置参数与来源算法以报告 §8.1 所引原报告为准，不在本报告重复标注。", 0.35, 6.6, 12.6, 0.36, ORANGE);
  footerNum(s, 29);
}

// ================= 30. 12-bit HMWS 重采：位深假设证伪 =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "HMWS 12-BIT RE-SAMPLE", "12-bit HMWS 重采：Noll 4 位深假设被证伪");
  callout(s, "裁定：位深不是 Noll 4 失效原因。12-bit 重采后 Noll 4 反而更差（R² 0.2911），不作系数定标；Noll 5 仍可用，但须重新标定。", 0.55, 1.32, 12.25, 0.72, RED);
  const hd = (t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: NAVY } } });
  const bad = (t) => ({ text: t, options: { bold: true, color: ORANGE } });
  const ok = (t) => ({ text: t, options: { bold: true, color: TEAL } });
  const rows = [
    [hd("指标"), hd("Noll 4 · 8-bit"), hd("Noll 4 · 12-bit"), hd("Noll 5 · 8-bit"), hd("Noll 5 · 12-bit")],
    ["slope", "0.3947", bad("0.1772"), "1.0753", ok("0.8662")],
    ["R²", "0.9302", bad("0.2911"), "≈0.999996", ok("0.9726")],
    ["ΔRMSE (rad)", "0.1401", bad("0.2814"), "0.0536", ok("0.0997")],
    ["系数可用性", "参考，不采用", bad("不作系数定标"), "可用（首采口径）", ok("仍可用，需重新标定")],
  ];
  s.addTable(rows, { x: 0.55, y: 2.2, w: 12.25, colW: [1.75, 2.55, 2.65, 2.55, 2.75], rowH: 0.55, fontFace: FONT, fontSize: 12, color: NAVY, border: { type: "solid", color: "D5DEE8", pt: 1 }, fill: { color: WHITE }, valign: "mid" });
  callout(s, "质量门禁：6/6 valid · 0 clipped · 暗帧严格匹配 0.08 ms · 12-bit / full_scale=4095 · 曝光收敛 converged=true。", 0.55, 5.05, 12.25, 0.55, TEAL);
  callout(s, "下一步：排查离焦偏置 / 近焦面响应 / 光斑弥散与增益耦合；下次上机先确认 SLM 首读正常，再复核 Noll 4。", 0.55, 5.75, 12.25, 0.65, ORANGE);
  captureNote(s, "源：data/hmws_zernike_probe/20261126_053859Z/summary.json · probe_config.json（8-bit 对照取本日首采报告）", 0.55, 6.55, 12.25);
  footerNum(s, 30);
}

// ================= 31. 12-bit CCD 远场：三点注入实拍 =================
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "12-BIT CCD / FAR FIELD", "12-bit CCD 远场：Noll 4 / 5 三点注入");
  s.addText("暗减 a 帧 · 六张共同裁剪 512 px · 统一显示窗 gamma 0.45 / 0–2000 counts", { x: 0.6, y: 1.22, w: 12.1, h: 0.26, fontFace: FONT, fontSize: 10.5, color: GRAY });
  const panels = [
    { file: "hmws12_n04_m20.png", noll: 4, inject: "−0.2", rec: "0.2396", delta: "+0.4396" },
    { file: "hmws12_n04_p0.png",  noll: 4, inject: "0",    rec: "0.1792", delta: "+0.1792" },
    { file: "hmws12_n04_p20.png", noll: 4, inject: "+0.2", rec: "0.3105", delta: "+0.1105" },
    { file: "hmws12_n05_m20.png", noll: 5, inject: "−0.2", rec: "−0.2508", delta: "−0.0508" },
    { file: "hmws12_n05_p0.png",  noll: 5, inject: "0",    rec: "−0.1279", delta: "−0.1279" },
    { file: "hmws12_n05_p20.png", noll: 5, inject: "+0.2", rec: "0.0957", delta: "−0.1043" },
  ];
  panels.forEach((p, i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const x = 0.6 + col * 3.95;
    const y = 1.6 + row * 2.5;
    s.addImage({ path: path.join(FIG, p.file), x, y, w: 2.38, h: 2.38 });
    s.addText(`Noll ${p.noll} · 注入 ${p.inject} rad`, { x: x + 2.48, y: y + 0.12, w: 1.4, h: 0.4, fontFace: FONT, fontSize: 11, color: NAVY, bold: true });
    s.addText("recovered_rad", { x: x + 2.48, y: y + 0.72, w: 1.4, h: 0.25, fontFace: FONT, fontSize: 9.5, color: GRAY });
    s.addText(`${p.rec} rad`, { x: x + 2.48, y: y + 1.0, w: 1.4, h: 0.42, fontFace: FONT, fontSize: 14, color: p.noll === 4 ? ORANGE : TEAL, bold: true });
    s.addText(`Δ = ${p.delta} rad`, { x: x + 2.48, y: y + 1.5, w: 1.4, h: 0.25, fontFace: FONT, fontSize: 9.5, color: GRAY });
  });
  s.addShape(pptx.ShapeType.roundRect, { x: 0.55, y: 6.55, w: 12.25, h: 0.4, fill: { color: LIGHT }, line: { color: GRAY, width: 1 } });
  s.addText("采集 2026-11-26 13:40:54–13:45:29（UTC+8）· 0.08 ms · 12-bit · bias=0.3 rad · radius=450 px · dark-exp-0.08ms.npz\n暗减 a 帧 · 固定共同裁剪 512 px（中心 row 1103 / col 1227）· 统一显示窗；非全帧、非逐图归一化。", { x: 0.7, y: 6.57, w: 11.95, h: 0.38, fontFace: FONT, fontSize: 9, color: NAVY, valign: "mid" });
  footerNum(s, 31);
}

// ================= 32. 完整 sweep：四档 lr × 60 迭代实测远场 =================
// 与 P27 并存：P27 为另一会话的 20 帧时间戳版；本页为本轮 512 帧完整 sweep。
{
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  header(s, "ANIMATION / COMPLETE SWEEP", "完整实测序列：四档 lr × 60 迭代（2026-11-27）");
  // GIF 原始 560×886，按等比置于左侧；右侧放逐档指标，避免拉伸变形。
  s.addImage({ path: path.join(ROOT, "images", "lr_farfield_evolution_20261127_hw.gif"), x: 0.55, y: 1.45, w: 3.35, h: 5.30 });
  captureNote(s, "GIF：60 帧（iter 0–59）· 窗口 256 px · log10(1+I) · 整组共用灰度上限 p99.5 · 未暗减", 0.55, 6.78, 3.35);
  const hd = (t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: NAVY } } });
  const rows = [
    [hd("lr"), hd("loss 0→59"), hd("降幅"), hd("峰值"), hd("FWHM(px)")],
    ["0.02", "3.39e-4 → 1.63e-4", "52.0%", "57 → 50", "17.8 → 17.9"],
    ["0.05", "3.39e-4 → 1.48e-4", "56.4%", "54 → 48", "17.9 → 17.2"],
    ["0.08", "3.39e-4 → 1.51e-4", "55.6%", "49 → 49", "18.2 → 18.3"],
    ["0.12", "3.39e-4 → 1.60e-4", "52.8%", "54 → 48", "17.9 → 17.3"],
  ];
  s.addTable(rows, { x: 4.15, y: 1.45, w: 8.65, colW: [0.85, 2.65, 1.35, 1.85, 1.95], rowH: 0.52, fontFace: FONT, fontSize: 12, color: NAVY, border: { type: "solid", color: "D5DEE8", pt: 1 }, fill: { color: WHITE }, valign: "mid" });
  bullets(s, [
    "采集口径：源 run 20261116_114424 的 params / phase_initial / focus；4 档 lr 各 60 迭代 + 平相位参考，共 512 帧。",
    "曝光 0.05 ms、CCD 8-bit（回读）；512 帧全部落盘、零饱和、无缺帧，journal 逐帧记 peak / counts / FWHM。",
    "实测裁定：四档 lr 的模型 loss 降幅接近（52.0%–56.4%），未复现离线代理「0.05 最优、0.08/0.12 退化」的分离度。",
    "远场 FWHM 全程维持 17–18 px、峰值稳定或缓降；本页展示逐帧演化，均匀度判定仍以 §5 同曝光配对 CV 为准。",
  ], 4.15, 4.35, 8.65, 2.2, 11.5);
  callout(s, "溯源：data/debug/differentiable_opt/lr_farfield_hw_20261127_real（run_config.json + journal.jsonl + 512 npz）；逐帧指标 artifacts/lr_farfield_hw_metrics.csv；静态总览 figures/fig14_lr_farfield_hw_contact_sheet.png。", 0.55, 6.28, 12.25, 0.52, GRAY);
  footerNum(s, 32);
}

pptx.writeFile({ fileName: OUT }).then(() => {
  console.log("PPTX written:", OUT);
  // PowerPoint (COM) rejects a plain Presentations.Open for two pptxgenjs quirks
  // that repair mode silently fixes: a repeated p:cNvPr shape id (slide 4: text
  // and table both id=5) and illegal table-cell `a:tcPr anchor="mid"` (not a
  // member of ST_TextAnchoringType; `ctr` is the legal equivalent). The fixer
  // rewrites only those attributes, in place, keeping all content bytes.
  const { spawnSync } = require("child_process");
  const fixer = path.join(__dirname, "fix_pptx_open.py");
  const r = spawnSync("python", [fixer, OUT], { stdio: "inherit", encoding: "utf8" });
  if (r.error || r.status !== 0) {
    console.error("FAIL postprocess (pptx open-compat fix):", r.error || `exit ${r.status}`);
    process.exit(1);
  }
}).catch((e) => {
  console.error("FAIL", e);
  process.exit(1);
});
