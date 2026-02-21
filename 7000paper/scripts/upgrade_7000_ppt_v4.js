const fs = require("fs");

const inPath = "7000paper/ppt/7000_presentation_spec_v3.json";
const outPath = "7000paper/ppt/7000_presentation_spec_v4.json";

const spec = JSON.parse(fs.readFileSync(inPath, "utf8"));

function textEl(text, x, y, w, h, options = {}) {
  return {
    type: "text",
    x,
    y,
    w,
    h,
    options: {
      text,
      fontFace: "Microsoft YaHei",
      color: "5F718D",
      fontSize: 9.5,
      ...options,
    },
  };
}

function shapeEl(text, x, y, w, h, options = {}) {
  return {
    type: "shape",
    x,
    y,
    w,
    h,
    options: {
      type: "roundRect",
      fill: "FFFFFF",
      line: { color: "D4DEEA", width: 1 },
      text,
      fontFace: "Microsoft YaHei",
      color: "10243F",
      fontSize: 12,
      align: "left",
      valign: "top",
      ...options,
    },
  };
}

function ensureFooterBrand(slide) {
  const exists = (slide.elements || []).some(
    (e) =>
      e.type === "text" &&
      e.options &&
      typeof e.options.text === "string" &&
      e.options.text.includes("DSCI 7000 | 双口径评估")
  );
  if (!exists) {
    slide.elements.push(
      textEl("DSCI 7000 | 双口径评估", 0.45, 7.18, 3.8, 0.2, {
        align: "left",
        fontFace: "Calibri",
      })
    );
  }
}

// Slide 1 subtitle: draft -> final
{
  const slide = spec.slides[0];
  const subtitle = (slide.elements || []).find(
    (e) =>
      e.type === "text" &&
      e.options &&
      typeof e.options.text === "string" &&
      e.options.text.includes("DSCI 7000 Capstone")
  );
  if (subtitle) {
    subtitle.options.text = "DSCI 7000 Capstone | Final Presentation v4";
  }
}

// Add consistent brand footer for slides 2-10
for (let i = 1; i < spec.slides.length; i++) {
  ensureFooterBrand(spec.slides[i]);
}

// Slide 5: metric footnote under title band
{
  const slide = spec.slides[4];
  slide.elements.push(
    textEl(
      "指标释义：ABR=审批坏账率；Reject Rate=拒绝率；主榜比较在 RR≈35% 约束下进行。",
      0.55,
      0.72,
      11.8,
      0.17,
      { color: "DCE8FA", fontSize: 9, fontFace: "Calibri", align: "left" }
    )
  );
}

// Slide 6: lift footnote
{
  const slide = spec.slides[5];
  slide.elements.push(
    textEl("注：Lift = Precision / Base Bad Rate（基准坏账率）", 0.75, 6.95, 6.8, 0.2, {
      color: "5F718D",
      fontSize: 9.5,
      align: "left",
      fontFace: "Calibri",
    })
  );
}

// Slide 7: sharpen monitoring wording + add color cues
{
  const slide = spec.slides[6];
  const ruleBox = (slide.elements || []).find(
    (e) =>
      e.type === "shape" &&
      e.options &&
      typeof e.options.text === "string" &&
      e.options.text.startsWith("RR 监控规则")
  );
  if (ruleBox) {
    ruleBox.options.text =
      "RR 监控规则（按 |RR_gap| 分层）\n\n绿色：<= 0.01（稳定）\n黄色：(0.01, 0.03]（预警）\n红色：> 0.03（触发复标定）\n\nRR_gap = RR_actual - RR_target";
    ruleBox.options.fontSize = 12.8;
  }

  slide.elements.push({
    type: "shape",
    x: 0.95,
    y: 2.15,
    w: 0.22,
    h: 0.22,
    options: { type: "roundRect", fill: "8FCFB7", line: { color: "8FCFB7", width: 0 } },
  });
  slide.elements.push({
    type: "shape",
    x: 0.95,
    y: 2.67,
    w: 0.22,
    h: 0.22,
    options: { type: "roundRect", fill: "F2CD62", line: { color: "F2CD62", width: 0 } },
  });
  slide.elements.push({
    type: "shape",
    x: 0.95,
    y: 3.2,
    w: 0.22,
    h: 0.22,
    options: { type: "roundRect", fill: "E7847B", line: { color: "E7847B", width: 0 } },
  });
}

// Slide 8: convert roadmap to 30/60/90-day cadence
{
  const slide = spec.slides[7];
  const roadmap = (slide.elements || []).find(
    (e) =>
      e.type === "shape" &&
      e.options &&
      typeof e.options.text === "string" &&
      e.options.text.startsWith("后续路线")
  );
  if (roadmap) {
    roadmap.options.text =
      "30/60/90 天落地节奏\n\nD+30：完成时间外滚动验证与阈值回放\nD+60：上线 RR_gap 自动监控与告警\nD+90：推进结构化/文本联合建模试点\n\n里程碑门槛：稳定性达标后再评估 LLM 主判可行性";
    roadmap.options.fontSize = 12.8;
  }
}

// Slide 10: add two Q&A prompts for defense
{
  const slide = spec.slides[9];
  slide.elements.push(
    shapeEl("Q&A 引导\n\n1) 若业务追求更高召回，如何控制 RR 偏移？\n2) 何时让 LLM 从辅助层进入主判层？", 8.65, 6.15, 3.6, 1.1, {
      fill: "EEF3FB",
      line: { color: "A7C6E8", width: 1 },
      fontSize: 10.5,
      valign: "middle",
    })
  );
}

fs.mkdirSync("7000paper/ppt", { recursive: true });
fs.writeFileSync(outPath, JSON.stringify(spec, null, 2), "utf8");
console.log("Wrote", outPath, "slides=", spec.slides.length);
