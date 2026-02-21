const fs = require("fs");

const inPath = "7000paper/ppt/7000_presentation_spec_v2.json";
const outPath = "7000paper/ppt/7000_presentation_spec_v3.json";

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

function footer(text) {
  return {
    type: "text",
    x: 12.1,
    y: 7.18,
    w: 1,
    h: 0.2,
    options: {
      text,
      fontSize: 9,
      color: "5F718D",
      fontFace: "Calibri",
      align: "right",
    },
  };
}

// Slide 5: Protocol A redesigned
spec.slides[4] = {
  background: { color: "F4F7FB" },
  elements: [
    ...topBand("4. 口径 A 主结果（公平主榜）"),
    {
      type: "shape",
      x: 0.55,
      y: 0.9,
      w: 12.25,
      h: 0.3,
      options: {
        type: "roundRect",
        fill: "E8F0FC",
        line: { color: "E8F0FC", width: 0 },
        text: "主榜规则：仅比较 Reject Rate 接近 35% 的模型；扩展组仅用于策略观察。",
        fontSize: 11.5,
        fontFace: "Microsoft YaHei",
        color: "1C4E80",
        bold: true,
        align: "center",
        valign: "middle",
      },
    },
    {
      type: "table",
      x: 0.55,
      y: 1.3,
      w: 6.35,
      h: 3.7,
      options: {
        rows: [
          [
            { text: "模型", options: { bold: true, color: "FFFFFF", fill: "1C4E80" } },
            { text: "Precision", options: { bold: true, color: "FFFFFF", fill: "1C4E80" } },
            { text: "Recall", options: { bold: true, color: "FFFFFF", fill: "1C4E80" } },
            { text: "ABR", options: { bold: true, color: "FFFFFF", fill: "1C4E80" } },
          ],
          [{ text: "DataAnalysis", options: { fill: "EEF7F1", bold: true } }, { text: "0.4175", options: { fill: "EEF7F1" } }, { text: "0.4317", options: { fill: "EEF7F1" } }, { text: "0.3120", options: { fill: "EEF7F1" } }],
          [{ text: "ML logistic_tabular", options: { fill: "EEF3FB", bold: true } }, { text: "0.4154", options: { fill: "EEF3FB" } }, { text: "0.4538", options: { fill: "EEF3FB" } }, { text: "0.3098", options: { fill: "EEF3FB" } }],
          ["ML xgboost_tabular", "0.4030", "0.4378", "0.3178"],
          ["LLM Two-Stage", "0.3653", "0.3675", "0.3420"],
        ],
        fontSize: 11,
        fontFace: "Calibri",
        color: "10243F",
        border: { pt: 1, color: "C9D4E3" },
      },
    },
    {
      type: "chart",
      x: 7.15,
      y: 1.3,
      w: 5.6,
      h: 3.7,
      options: {
        type: "bar",
        data: [
          {
            name: "Precision",
            labels: ["DA", "L-Tab", "XGB", "LLM-2S"],
            values: [0.4175, 0.4154, 0.403, 0.3653],
          },
          {
            name: "Recall",
            labels: ["DA", "L-Tab", "XGB", "LLM-2S"],
            values: [0.4317, 0.4538, 0.4378, 0.3675],
          },
        ],
        showLegend: true,
        legendPos: "b",
      },
    },
    {
      type: "shape",
      x: 0.55,
      y: 5.15,
      w: 6.35,
      h: 1.8,
      options: {
        type: "roundRect",
        fill: "FFFFFF",
        line: { color: "8FCFB7", width: 1.3 },
        text: "关键观察：\nlogistic_tabular 的违约捕获率最高（Recall 0.4538），\nDataAnalysis 的 Precision 略高（0.4175）。",
        fontSize: 12.5,
        fontFace: "Microsoft YaHei",
        color: "10243F",
        align: "left",
        valign: "top",
      },
    },
    {
      type: "shape",
      x: 7.15,
      y: 5.15,
      w: 5.6,
      h: 1.8,
      options: {
        type: "roundRect",
        fill: "FFF8F7",
        line: { color: "E8B9B4", width: 1.3 },
        text: "扩展组提醒：\nOne-Stage LLM 与 GPT-5.2 的高召回伴随高 RR 偏移，\n不进入公平主榜结论。",
        fontSize: 12.5,
        fontFace: "Microsoft YaHei",
        color: "10243F",
        align: "left",
        valign: "top",
      },
    },
    footer("P5"),
  ],
};

// Slide 6: Protocol B redesigned
spec.slides[5] = {
  background: { color: "F4F7FB" },
  elements: [
    ...topBand("5. 口径 B 结果与误差结构"),
    {
      type: "shape",
      x: 0.75,
      y: 0.95,
      w: 5.95,
      h: 0.7,
      options: {
        type: "roundRect",
        fill: "EEF3FB",
        line: { color: "A7C6E8", width: 1.2 },
        text: "口径B最优：ML xgboost_tabular（Precision 0.3153, Lift 2.0568）",
        fontSize: 11.5,
        fontFace: "Microsoft YaHei",
        color: "10243F",
        bold: true,
        align: "center",
        valign: "middle",
      },
    },
    {
      type: "shape",
      x: 7.05,
      y: 0.95,
      w: 5.7,
      h: 0.7,
      options: {
        type: "roundRect",
        fill: "F3FAF7",
        line: { color: "8FCFB7", width: 1.2 },
        text: "排序变化：口径A与口径B最优模型不同，说明分布与目标函数会改变结论。",
        fontSize: 11.5,
        fontFace: "Microsoft YaHei",
        color: "10243F",
        bold: true,
        align: "center",
        valign: "middle",
      },
    },
    {
      type: "table",
      x: 0.75,
      y: 1.85,
      w: 6.0,
      h: 2.45,
      options: {
        rows: [
          [
            { text: "模型（口径B）", options: { bold: true, color: "FFFFFF", fill: "1C4E80" } },
            { text: "Precision", options: { bold: true, color: "FFFFFF", fill: "1C4E80" } },
            { text: "Recall", options: { bold: true, color: "FFFFFF", fill: "1C4E80" } },
            { text: "Lift", options: { bold: true, color: "FFFFFF", fill: "1C4E80" } },
          ],
          ["ML logistic_tabular", "0.3069", "0.2992", "2.0023"],
          [{ text: "ML xgboost_tabular", options: { fill: "EEF7F1", bold: true } }, { text: "0.3153", options: { fill: "EEF7F1" } }, { text: "0.3095", options: { fill: "EEF7F1" } }, { text: "2.0568", options: { fill: "EEF7F1" } }],
          ["DataAnalysis", "0.2849", "0.2851", "1.8589"],
        ],
        fontSize: 11,
        fontFace: "Calibri",
        color: "10243F",
        border: { pt: 1, color: "C9D4E3" },
      },
    },
    {
      type: "table",
      x: 7.05,
      y: 1.85,
      w: 5.7,
      h: 2.45,
      options: {
        rows: [
          [
            { text: "模型（口径A）", options: { bold: true, color: "FFFFFF", fill: "1C4E80" } },
            { text: "TP/1000", options: { bold: true, color: "FFFFFF", fill: "1C4E80" } },
            { text: "FN/1000", options: { bold: true, color: "FFFFFF", fill: "1C4E80" } },
            { text: "FP/1000", options: { bold: true, color: "FFFFFF", fill: "1C4E80" } },
          ],
          ["ML logistic_tabular", "158.9", "191.3", "223.6"],
          ["DataAnalysis", "151.2", "199.0", "211.0"],
          ["LLM Two-Stage", "128.7", "221.5", "223.6"],
        ],
        fontSize: 11,
        fontFace: "Calibri",
        color: "10243F",
        border: { pt: 1, color: "C9D4E3" },
      },
    },
    {
      type: "shape",
      x: 0.75,
      y: 4.55,
      w: 12.0,
      h: 2.35,
      options: {
        type: "roundRect",
        fill: "FFFFFF",
        line: { color: "D4DEEA", width: 1 },
        text:
          "业务解读：\n- 口径B中 xgboost_tabular 在真实分布更占优；口径A中 logistic_tabular 与 DataAnalysis 更稳健。\n- 说明“公平横评结论”与“部署口径结论”需要并列汇报，不应混用。",
        fontSize: 13,
        fontFace: "Microsoft YaHei",
        color: "10243F",
        align: "left",
        valign: "top",
      },
    },
    footer("P6"),
  ],
};

fs.writeFileSync(outPath, JSON.stringify(spec, null, 2), "utf8");
console.log("Wrote", outPath, "slides=", spec.slides.length);
