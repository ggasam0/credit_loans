const fs = require("fs");

const inPath = "7000paper/ppt/7000_presentation_spec_v1.json";
const outPath = "7000paper/ppt/7000_presentation_spec_v2.json";

const spec = JSON.parse(fs.readFileSync(inPath, "utf8"));

function topBand(titleText) {
  return [
    {
      type: "shape",
      x: 0,
      y: 0,
      w: 13.33,
      h: 0.7,
      options: { type: "rect", fill: "10243F" },
    },
    {
      type: "text",
      x: 0.45,
      y: 0.15,
      w: 11.0,
      h: 0.34,
      options: {
        text: titleText,
        fontSize: 20,
        bold: true,
        color: "FFFFFF",
        fontFace: "Microsoft YaHei",
      },
    },
  ];
}

function footer(page) {
  return {
    type: "text",
    x: 12.1,
    y: 7.18,
    w: 1,
    h: 0.2,
    options: {
      text: `P${page}`,
      fontSize: 9,
      color: "5F718D",
      fontFace: "Calibri",
      align: "right",
    },
  };
}

function slideModelFamily() {
  return {
    background: { color: "F4F7FB" },
    elements: [
      ...topBand("3. 模型家族与试点产物"),
      {
        type: "shape",
        x: 0.7,
        y: 1.15,
        w: 2.9,
        h: 4.8,
        options: {
          type: "roundRect",
          fill: "F3FAF7",
          line: { color: "8FCFB7", width: 1.2 },
          text: "DataAnalysis\n\n规则化风险画像\n可解释业务基线",
          fontSize: 13,
          fontFace: "Microsoft YaHei",
          color: "10243F",
          align: "left",
          valign: "top",
        },
      },
      {
        type: "shape",
        x: 3.9,
        y: 1.15,
        w: 2.9,
        h: 4.8,
        options: {
          type: "roundRect",
          fill: "EEF3FB",
          line: { color: "A7C6E8", width: 1.2 },
          text: "ML\n\nlogistic\nxgboost",
          fontSize: 13,
          fontFace: "Microsoft YaHei",
          color: "10243F",
          align: "left",
          valign: "top",
        },
      },
      {
        type: "shape",
        x: 7.1,
        y: 1.15,
        w: 2.7,
        h: 4.8,
        options: {
          type: "roundRect",
          fill: "F8F5FD",
          line: { color: "CDB8E8", width: 1.2 },
          text: "BERT\n\nEmbedding\nFinetune",
          fontSize: 13,
          fontFace: "Microsoft YaHei",
          color: "10243F",
          align: "left",
          valign: "top",
        },
      },
      {
        type: "shape",
        x: 10.1,
        y: 1.15,
        w: 2.5,
        h: 4.8,
        options: {
          type: "roundRect",
          fill: "FDF5F4",
          line: { color: "E7B8B1", width: 1.2 },
          text: "LLM\n\nTwo-stage\nOne-stage\nGPT-5.2",
          fontSize: 13,
          fontFace: "Microsoft YaHei",
          color: "10243F",
          align: "left",
          valign: "top",
        },
      },
      {
        type: "shape",
        x: 0.7,
        y: 6.1,
        w: 11.9,
        h: 0.85,
        options: {
          type: "roundRect",
          fill: "DDE8F5",
          line: { color: "DDE8F5", width: 0 },
          text: "试点通过标准：数据一致、流程一致、结果可回查、偏差可解释（Pilot Exit Criteria）",
          fontSize: 12.5,
          fontFace: "Microsoft YaHei",
          bold: true,
          color: "1C4E80",
          align: "center",
          valign: "middle",
        },
      },
      footer(4),
    ],
  };
}

function slideRules() {
  return {
    background: { color: "F4F7FB" },
    elements: [
      ...topBand("6. 监控规则与双榜汇报边界"),
      {
        type: "shape",
        x: 0.7,
        y: 1.15,
        w: 6.0,
        h: 2.8,
        options: {
          type: "roundRect",
          fill: "FFFFFF",
          line: { color: "D4DEEA", width: 1 },
          text:
            "RR 监控规则\n\nRR_gap = RR_actual - RR_target\n|RR_gap| <= 0.01：稳定\n0.01 < |RR_gap| <= 0.03：预警\n|RR_gap| > 0.03：触发复标定",
          fontSize: 13.5,
          fontFace: "Microsoft YaHei",
          color: "10243F",
          align: "left",
          valign: "top",
        },
      },
      {
        type: "shape",
        x: 0.7,
        y: 4.2,
        w: 6.0,
        h: 2.7,
        options: {
          type: "roundRect",
          fill: "FFFFFF",
          line: { color: "D4DEEA", width: 1 },
          text:
            "口径A示例（RR_target=35%）\n\nDataAnalysis：RR_actual=0.3622，RR_gap=+0.0122\nlogistic_tabular：RR_gap=+0.0326\nOne-stage LLM：RR_gap=+0.2597\nGPT-5.2：RR_gap=+0.4728",
          fontSize: 13,
          fontFace: "Microsoft YaHei",
          color: "10243F",
          align: "left",
          valign: "top",
        },
      },
      {
        type: "shape",
        x: 7.05,
        y: 1.15,
        w: 5.7,
        h: 5.75,
        options: {
          type: "roundRect",
          fill: "FFFFFF",
          line: { color: "D4DEEA", width: 1 },
          text:
            "主榜/扩展榜双层规则\n\n主榜（公平比较）：\n- 仅纳入 RR 接近目标值模型\n- 比较 Precision / Lift / ABR\n\n扩展榜（策略观察）：\n- 报告所有模型\n- 解释“高召回是否由高拒绝率驱动”\n\n结论使用原则：\n主榜决定上线候选，扩展榜用于监控与校准设计",
          fontSize: 13,
          fontFace: "Microsoft YaHei",
          color: "10243F",
          align: "left",
          valign: "top",
        },
      },
      footer(7),
    ],
  };
}

function slideEvidence() {
  return {
    background: { color: "F4F7FB" },
    elements: [
      ...topBand("8. 可复现证据链与交付清单"),
      {
        type: "table",
        x: 0.75,
        y: 1.15,
        w: 12.0,
        h: 3.25,
        options: {
          rows: [
            [
              { text: "环节", options: { bold: true, color: "FFFFFF", fill: "1C4E80" } },
              { text: "关键产物", options: { bold: true, color: "FFFFFF", fill: "1C4E80" } },
              { text: "审计检查点", options: { bold: true, color: "FFFFFF", fill: "1C4E80" } },
            ],
            ["数据准备", "shared_subset / full_processed", "样本规模、标签口径一致"],
            ["模型训练", "run_*.py + split files", "是否仅 train/validation 定阈"],
            ["测试评估", "run_report.json", "是否存在测试集反向调参"],
            ["报告固化", "7000main.md / presentation", "表格数值与JSON一致"],
          ],
          fontSize: 12,
          fontFace: "Microsoft YaHei",
          color: "10243F",
          border: { pt: 1, color: "C9D4E3" },
        },
      },
      {
        type: "shape",
        x: 0.75,
        y: 4.65,
        w: 12.0,
        h: 2.25,
        options: {
          type: "roundRect",
          fill: "FFFFFF",
          line: { color: "D4DEEA", width: 1 },
          text:
            "最终提交前检查清单\n\n1) 研究问题与假说在结果章节有直接证据\n2) 无测试集调参，阈值流程一致\n3) 主文档结论可回查到 run_report.json\n4) 展示稿与书面报告叙述一致",
          fontSize: 13.5,
          fontFace: "Microsoft YaHei",
          color: "10243F",
          align: "left",
          valign: "top",
        },
      },
      footer(9),
    ],
  };
}

const slides = spec.slides.slice();

// Insert new slide after current "2. 双口径方法设计"
slides.splice(3, 0, slideModelFamily());

// Insert monitoring/rules slide after current "4. 口径 B 结果与误差结构"
const idxB = slides.findIndex((s) =>
  (s.elements || []).some(
    (e) => e.type === "text" && e.options && String(e.options.text).includes("口径 B 结果与误差结构")
  )
);
if (idxB >= 0) {
  slides.splice(idxB + 1, 0, slideRules());
}

// Insert evidence slide before final conclusion slide
const idxConclusion = slides.findIndex((s) =>
  (s.elements || []).some(
    (e) => e.type === "text" && e.options && String(e.options.text).includes("结论与 Q&A")
  )
);
if (idxConclusion >= 0) {
  slides.splice(idxConclusion, 0, slideEvidence());
}

// Renumber existing section titles
const renumberMap = [
  ["3. 口径 A 主结果（公平主榜）", "4. 口径 A 主结果（公平主榜）"],
  ["4. 口径 B 结果与误差结构", "5. 口径 B 结果与误差结构"],
  ["5. 部署建议与下一步", "7. 部署建议与下一步"],
  ["6. 结论与 Q&A", "9. 结论与 Q&A"],
];

slides.forEach((slide) => {
  (slide.elements || []).forEach((el) => {
    if (el.type === "text" && el.options && typeof el.options.text === "string") {
      const oldText = el.options.text;
      for (const [from, to] of renumberMap) {
        if (oldText === from) {
          el.options.text = to;
        }
      }
      if (oldText.startsWith("汇报目录")) {
        el.options.text =
          "汇报目录\n\n1) 双口径评估协议\n2) 模型家族与试点产物\n3) 口径A主结果（公平主榜）\n4) 口径B部署结果（全量分布）\n5) RR_gap监控与双榜规则\n6) 部署建议与后续路线\n7) 可复现证据链\n8) 结论与Q&A";
      }
      if (/^P\d+$/.test(oldText)) {
        // reset later
        el.options.text = "P?";
      }
    }
  });
});

// Normalize page footers after final ordering
for (let i = 1; i < slides.length; i++) {
  const target = `P${i + 1}`;
  let found = false;
  (slides[i].elements || []).forEach((el) => {
    if (
      el.type === "text" &&
      el.options &&
      typeof el.options.text === "string" &&
      /^P\?$/i.test(el.options.text)
    ) {
      el.options.text = target;
      found = true;
    }
  });
  if (!found) {
    slides[i].elements.push(footer(i + 1));
  }
}

spec.slides = slides;

fs.mkdirSync("7000paper/ppt", { recursive: true });
fs.writeFileSync(outPath, JSON.stringify(spec, null, 2), "utf8");
console.log("Wrote", outPath, "slides=", slides.length);
