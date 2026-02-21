/*
 * Generate a presentation deck for 6800main.md using PptxGenJS.
 * Usage:
 *   npx -y -p pptxgenjs node 6800paper/scripts/md_to_pptx_6800.js \
 *     6800paper/6800_final_presentation.pptx 6800paper/6800main.md
 */

const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");

const outArg = process.argv[2];
const mdArg = process.argv[3];

const outFile =
  outArg || path.resolve(__dirname, "..", "6800_final_presentation.pptx");
const mdFile = mdArg || path.resolve(__dirname, "..", "6800main.md");

const mdText = fs.readFileSync(mdFile, "utf8");

function pick(re, fallback) {
  const m = mdText.match(re);
  return m ? m[1].trim() : fallback;
}

const title = pick(/^#\s+(.+)$/m, "6800 Final Project");
const author = pick(/^作者：\s*(.+)$/m, "<姓名/学号>");
const course = pick(/^课程：\s*(.+)$/m, "6800 Final Project");
const dateText = pick(/^日期：\s*(.+)$/m, "2026-02-21");

const theme = {
  navy: "10243F",
  blue: "1C4E80",
  teal: "2FA4A6",
  light: "F4F7FB",
  text: "10243F",
  muted: "4B5D78",
  white: "FFFFFF",
};

const font = {
  zh: "Microsoft YaHei",
  en: "Calibri",
};

function baseSlide(pptx) {
  const s = pptx.addSlide();
  s.background = { color: theme.light };
  return s;
}

function addTopBand(slide, sectionTitle) {
  slide.addShape(slide.pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: 13.33,
    h: 0.72,
    fill: { color: theme.navy },
    line: { color: theme.navy, pt: 0 },
  });
  slide.addText(sectionTitle, {
    x: 0.45,
    y: 0.16,
    w: 10.6,
    h: 0.36,
    fontFace: font.zh,
    fontSize: 19,
    bold: true,
    color: theme.white,
    margin: 0,
  });
  slide.addText("6800 Final", {
    x: 11.3,
    y: 0.2,
    w: 1.6,
    h: 0.28,
    fontFace: font.en,
    fontSize: 12,
    bold: true,
    color: "A8C4E8",
    align: "right",
    margin: 0,
  });
}

function addFooter(slide, page) {
  slide.addText(`第 ${page} 页`, {
    x: 12.15,
    y: 7.14,
    w: 1.0,
    h: 0.2,
    fontFace: font.zh,
    fontSize: 9,
    color: "5F718D",
    align: "right",
    margin: 0,
  });
}

function addCard(slide, x, y, w, h, heading, body) {
  slide.addShape(slide.pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.08,
    fill: { color: "FFFFFF" },
    line: { color: "D4DEEA", pt: 1 },
    shadow: { type: "outer", color: "000000", blur: 2, offset: 1, angle: 45, opacity: 0.08 },
  });
  slide.addText(heading, {
    x: x + 0.18,
    y: y + 0.14,
    w: w - 0.36,
    h: 0.36,
    fontFace: font.zh,
    fontSize: 16,
    bold: true,
    color: theme.blue,
    margin: 0,
  });
  slide.addText(body, {
    x: x + 0.18,
    y: y + 0.56,
    w: w - 0.36,
    h: h - 0.72,
    fontFace: font.zh,
    fontSize: 12.5,
    color: theme.text,
    valign: "top",
    margin: 0,
  });
}

async function build() {
  const pptx = new PptxGenJS();
  pptx.layout = "LAYOUT_WIDE";
  pptx.author = author;
  pptx.company = "6800 Course Project";
  pptx.subject = "Credit Risk Modeling with Text Features";
  pptx.title = title;
  pptx.lang = "zh-CN";

  // Slide 1: Cover
  {
    const s = pptx.addSlide();
    s.background = { color: theme.navy };
    s.addShape(s.pptx.ShapeType.rect, {
      x: 0,
      y: 0,
      w: 13.33,
      h: 0.28,
      fill: { color: theme.teal },
      line: { color: theme.teal, pt: 0 },
    });
    s.addShape(s.pptx.ShapeType.roundRect, {
      x: 0.7,
      y: 1.05,
      w: 12.0,
      h: 4.65,
      rectRadius: 0.08,
      fill: { color: "173155" },
      line: { color: "366A9F", pt: 1 },
    });
    s.addText(title, {
      x: 1.05,
      y: 1.5,
      w: 11.3,
      h: 1.8,
      fontFace: font.zh,
      fontSize: 38,
      bold: true,
      color: theme.white,
      align: "left",
      valign: "top",
      margin: 0,
    });
    s.addText("传统模型与两阶段 LLM 的统一口径实验报告", {
      x: 1.05,
      y: 3.5,
      w: 11.0,
      h: 0.55,
      fontFace: font.zh,
      fontSize: 19,
      color: "D4E6FF",
      margin: 0,
    });
    s.addText(`${author}    |    ${course}    |    ${dateText}`, {
      x: 1.05,
      y: 5.0,
      w: 11.0,
      h: 0.36,
      fontFace: font.zh,
      fontSize: 13,
      color: "A8C4E8",
      margin: 0,
    });
    s.addText("Focus: Fixed Reject Rate fairness, reproducibility, and deployability", {
      x: 1.05,
      y: 6.45,
      w: 11.4,
      h: 0.4,
      fontFace: font.en,
      fontSize: 13,
      italic: true,
      color: "89A9D2",
      margin: 0,
    });
  }

  // Slide 2: Motivation and contributions
  {
    const s = baseSlide(pptx);
    addTopBand(s, "1. 研究动机与问题定义");
    addCard(
      s,
      0.7,
      1.05,
      6.0,
      5.9,
      "研究背景",
      "• P2P 借贷存在显著信息不对称，结构化变量不能完全覆盖偿付风险。\n" +
        "• 借款描述文本（desc）包含还款意愿、财务压力、用途可信度等软信息。\n" +
        "• 仅看离线 AUC 往往与实际审批策略脱节，难支持上线决策。"
    );
    addCard(
      s,
      6.95,
      1.05,
      5.65,
      5.9,
      "本文核心问题与贡献",
      "RQ1: 固定拒绝率约束下，文本模型能否稳定优于结构化基线？\n" +
        "RQ2: 拒绝率目标变化时，模型排序是否迁移？\n" +
        "RQ3: 两阶段 LLM 的阈值重标定能否转化为业务增益？\n\n" +
        "贡献：\n" +
        "1) 统一代码与口径的公平横评框架。\n" +
        "2) 共享子集 + 全量分布双口径实证。\n" +
        "3) 提供阈值敏感性、RR 扫描与置信区间。"
    );
    addFooter(s, 2);
  }

  // Slide 3: Literature synthesis
  {
    const s = baseSlide(pptx);
    addTopBand(s, "2. 文献脉络与研究定位");

    const stages = [
      ["经典评分", "建立信用评分评估范式\n(统计分类、行为评分)"],
      ["P2P 信息不对称", "证明软筛选存在\n文本/社交信息有增量价值"],
      ["文本特征工程", "loan title / desc\n显著提升违约识别能力"],
      ["深度文本模型", "上下文建模能力增强\n但解释与校准仍是难点"],
      ["LLM 阶段", "语义能力更强\n落地核心转向阈值与稳定性"],
    ];

    let x = 0.55;
    for (const [h, b] of stages) {
      addCard(s, x, 1.35, 2.45, 4.8, h, b);
      x += 2.55;
    }

    s.addText(
      "本文定位：不追求“最高离线分”，而是验证同口径、同阈值策略下的可复现业务增益。",
      {
        x: 0.8,
        y: 6.45,
        w: 12.0,
        h: 0.52,
        fontFace: font.zh,
        fontSize: 14,
        bold: true,
        color: theme.blue,
        align: "center",
        margin: 0,
      }
    );
    addFooter(s, 3);
  }

  // Slide 4: Data and label setup
  {
    const s = baseSlide(pptx);
    addTopBand(s, "3. 数据集与任务定义");

    const rows = [
      ["阶段", "样本量", "说明"],
      ["原始读取", "2,260,701", "accepted_2007_to_2018Q4.csv"],
      ["初筛后", "123,293", "满足基础状态与字段条件"],
      ["清洗后", "123,202", "去重/缺失处理后可用样本"],
      ["共享高风险子集", "7,108", "grade∈{E,F,G}, int_rate>=18.5, annual_inc<=92,500"],
    ];
    s.addTable(rows, {
      x: 0.75,
      y: 1.15,
      w: 12.0,
      h: 2.65,
      fontFace: font.zh,
      fontSize: 11.5,
      border: { pt: 1, color: "C9D4E3" },
      color: theme.text,
      fill: "FFFFFF",
      valign: "middle",
      margin: 0.05,
    });

    addCard(
      s,
      0.75,
      4.05,
      5.95,
      2.8,
      "标签与任务",
      "• target=1: Charged Off（违约）\n" +
        "• target=0: Fully Paid（履约）\n" +
        "• 评估目标不是单纯分类精度，而是固定拒绝率下的业务风险识别质量。"
    );
    addCard(
      s,
      6.95,
      4.05,
      5.8,
      2.8,
      "双口径设计",
      "口径 A（公平横评）：共享子集 + RR=35%\n" +
        "口径 B（业务分布）：全量样本 + RR=0.1533\n\n" +
        "用途：分离“模型能力”与“样本分布/策略口径”影响。"
    );
    addFooter(s, 4);
  }

  // Slide 5: Methods
  {
    const s = baseSlide(pptx);
    addTopBand(s, "4. 方法框架与实现路线");

    addCard(s, 0.65, 1.35, 3.0, 4.8, "DataAnalysis", "规则化风险画像\n验证集定阈\n低复杂度、可解释基线");
    addCard(s, 3.95, 1.35, 3.0, 4.8, "传统 ML", "logistic_tabular\nlogistic_text_fusion\nxgboost_tabular");
    addCard(s, 7.25, 1.35, 2.8, 4.8, "BERT", "Embedding + LR\n端到端微调\nmax_length=512");
    addCard(s, 10.3, 1.35, 2.35, 4.8, "LLM", "Two-stage\nOne-stage\nExternal GPT-5.2");

    s.addShape(s.pptx.ShapeType.chevron, {
      x: 0.95,
      y: 6.35,
      w: 11.95,
      h: 0.5,
      fill: { color: "DDE8F5" },
      line: { color: "DDE8F5", pt: 0 },
    });
    s.addText("统一评估协议：验证集定阈 -> 测试集固定评估（禁止测试集调参）", {
      x: 1.1,
      y: 6.43,
      w: 11.3,
      h: 0.3,
      fontFace: font.zh,
      fontSize: 12.5,
      color: theme.blue,
      bold: true,
      margin: 0,
      align: "center",
    });
    addFooter(s, 5);
  }

  // Slide 6: Metrics and protocol
  {
    const s = baseSlide(pptx);
    addTopBand(s, "5. 实验协议与指标体系");
    addCard(
      s,
      0.75,
      1.1,
      5.9,
      3.0,
      "公平比较控制",
      "• 统一切分文件与 _shared_row_id 对齐。\n" +
        "• 统一阈值流程：验证集按目标 RR 取阈值。\n" +
        "• 拒绝率偏离显著的模型不并入公平主榜。"
    );
    addCard(
      s,
      6.9,
      1.1,
      5.7,
      3.0,
      "业务指标（Reject 为正类）",
      "Precision@RR = TP/(TP+FP)\n" +
        "Recall@RR = TP/(TP+FN)\n" +
        "Approval Bad Rate = FN/(TN+FN)\n" +
        "Lift@RR = Precision@RR / RR_target"
    );

    const protocolRows = [
      ["实验阶段", "目的"],
      ["主对比（口径A）", "在 RR=35% 约束下做公平横评"],
      ["扩展 LLM 对比", "分析高召回是否来自高拒绝率策略"],
      ["RR 扫描 + Bootstrap CI", "检验策略迁移性与统计稳健性"],
      ["全量口径B", "验证在真实分布下排序是否变化"],
    ];
    s.addTable(protocolRows, {
      x: 0.75,
      y: 4.35,
      w: 11.85,
      h: 2.35,
      fontFace: font.zh,
      fontSize: 12,
      border: { pt: 1, color: "C9D4E3" },
      fill: "FFFFFF",
      color: theme.text,
      margin: 0.05,
    });
    addFooter(s, 6);
  }

  // Slide 7: Main results A
  {
    const s = baseSlide(pptx);
    addTopBand(s, "6. 主结果（口径A：共享子集，RR=35%）");

    const tableA = [
      ["模型", "Precision@RR", "Recall@RR", "Approval Bad Rate", "Lift@RR", "Reject Rate"],
      ["DataAnalysis", "0.4175", "0.4317", "0.3120", "1.1928", "0.3622"],
      ["logistic_tabular", "0.4154", "0.4538", "0.3098", "1.1870", "0.3826"],
      ["logistic_text_fusion", "0.4154", "0.4337", "0.3126", "1.1868", "0.3657"],
      ["xgboost_tabular", "0.4030", "0.4378", "0.3178", "1.1513", "0.3805"],
      ["BERT-Embedding", "0.3859", "0.4177", "0.3284", "1.1026", "0.3790"],
      ["BERT-Finetune", "0.3895", "0.4317", "0.3253", "1.1128", "0.3882"],
      ["LLM Two-Stage", "0.3653", "0.3675", "0.3420", "1.0436", "0.3523"],
    ];
    s.addTable(tableA, {
      x: 0.45,
      y: 1.0,
      w: 12.45,
      h: 3.95,
      fontFace: font.en,
      fontSize: 10.5,
      border: { pt: 1, color: "C9D4E3" },
      fill: "FFFFFF",
      color: theme.text,
      margin: 0.04,
      valign: "middle",
    });

    s.addChart(pptx.ChartType.bar, [
      { name: "Precision@RR", labels: ["DA", "L-Tab", "L-Fusion", "XGB", "BERT-FT", "LLM-2S"], values: [0.4175, 0.4154, 0.4154, 0.4030, 0.3895, 0.3653] },
    ], {
      x: 0.7,
      y: 5.1,
      w: 6.25,
      h: 1.95,
      barDir: "col",
      showLegend: false,
      catAxisLabelRotate: -45,
      valAxisMinVal: 0.34,
      valAxisMaxVal: 0.43,
      valAxisMajorUnit: 0.02,
      chartColors: [theme.blue],
    });

    addCard(
      s,
      7.2,
      5.06,
      5.65,
      1.98,
      "结论",
      "在公平 RR 约束下，DataAnalysis 与 logistic_tabular 最稳健。\n" +
        "LLM 两阶段可用但仍存在明显精度与放行坏账率差距。"
    );
    addFooter(s, 7);
  }

  // Slide 8: Extended LLM + full distribution
  {
    const s = baseSlide(pptx);
    addTopBand(s, "7. 扩展结果：LLM 与全量口径");

    s.addText("A) 扩展 LLM 对比（同测试集，未全部满足 RR 对齐）", {
      x: 0.75,
      y: 1.0,
      w: 6.2,
      h: 0.3,
      fontFace: font.zh,
      fontSize: 13,
      bold: true,
      color: theme.blue,
      margin: 0,
    });
    s.addTable(
      [
        ["模型", "Precision", "Recall", "Reject Rate"],
        ["LLM Two-Stage", "0.3653", "0.3675", "0.3523"],
        ["LLM One-Stage", "0.3587", "0.6245", "0.6097"],
        ["GPT-5.2 External", "0.3462", "0.8133", "0.8228"],
      ],
      {
        x: 0.75,
        y: 1.35,
        w: 6.0,
        h: 2.05,
        fontFace: font.en,
        fontSize: 11,
        border: { pt: 1, color: "C9D4E3" },
        fill: "FFFFFF",
        color: theme.text,
        margin: 0.05,
      }
    );
    addCard(
      s,
      0.75,
      3.55,
      6.0,
      2.5,
      "解释",
      "单阶段 LLM 和 GPT-5.2 的高召回主要由高拒绝率驱动，\n不属于同约束公平优势，故不纳入主榜结论。"
    );

    s.addText("B) 全量业务口径（n=123,202，RR=0.1533）", {
      x: 7.05,
      y: 1.0,
      w: 5.7,
      h: 0.3,
      fontFace: font.zh,
      fontSize: 13,
      bold: true,
      color: theme.blue,
      margin: 0,
    });
    s.addTable(
      [
        ["模型", "Precision", "Recall", "Lift"],
        ["logistic_tabular", "0.3069", "0.2992", "2.0023"],
        ["logistic_text_fusion", "0.2994", "0.3050", "1.9532"],
        ["xgboost_tabular", "0.3153", "0.3095", "2.0568"],
        ["DataAnalysis", "0.2849", "0.2851", "1.8589"],
      ],
      {
        x: 7.05,
        y: 1.35,
        w: 5.7,
        h: 2.35,
        fontFace: font.en,
        fontSize: 11,
        border: { pt: 1, color: "C9D4E3" },
        fill: "FFFFFF",
        color: theme.text,
        margin: 0.05,
      }
    );
    addCard(
      s,
      7.05,
      3.95,
      5.7,
      2.1,
      "结论",
      "样本分布回到全量后，xgboost_tabular 成为综合最优。\n模型排序并非固定，受 RR 目标与分布设定影响。"
    );
    addFooter(s, 8);
  }

  // Slide 9: Stability and business cost
  {
    const s = baseSlide(pptx);
    addTopBand(s, "8. 稳健性与业务代价解释");
    s.addText("拒绝率扫描（Precision）", {
      x: 0.75,
      y: 1.0,
      w: 5.0,
      h: 0.3,
      fontFace: font.zh,
      fontSize: 13,
      bold: true,
      color: theme.blue,
      margin: 0,
    });
    s.addChart(
      pptx.ChartType.line,
      [
        { name: "logistic_tabular", labels: ["10%", "20%", "35%", "50%"], values: [0.5180, 0.4712, 0.4154, 0.4023] },
        { name: "DataAnalysis", labels: ["10%", "20%", "35%", "50%"], values: [0.4797, 0.4254, 0.4175, 0.3992] },
        { name: "xgboost_tabular", labels: ["10%", "20%", "35%", "50%"], values: [0.4248, 0.4415, 0.4030, 0.3963] },
        { name: "BERT-FT", labels: ["10%", "20%", "35%", "50%"], values: [0.4044, 0.3789, 0.3895, 0.3826] },
      ],
      {
        x: 0.7,
        y: 1.35,
        w: 6.1,
        h: 3.0,
        showLegend: true,
        legendPos: "b",
        chartColors: [theme.blue, theme.teal, "637EA5", "9AAFD0"],
        valAxisMinVal: 0.36,
        valAxisMaxVal: 0.54,
      }
    );

    s.addText("每千笔申请误差结构（口径A）", {
      x: 7.05,
      y: 1.0,
      w: 5.6,
      h: 0.3,
      fontFace: font.zh,
      fontSize: 13,
      bold: true,
      color: theme.blue,
      margin: 0,
    });
    s.addTable(
      [
        ["模型", "TP/1000", "FN/1000", "FP/1000"],
        ["logistic_tabular", "158.9", "191.3", "223.6"],
        ["DataAnalysis", "151.2", "199.0", "211.0"],
        ["BERT-Finetune", "151.2", "199.0", "237.0"],
        ["LLM Two-Stage", "128.7", "221.5", "223.6"],
      ],
      {
        x: 7.05,
        y: 1.35,
        w: 5.65,
        h: 2.45,
        fontFace: font.en,
        fontSize: 11,
        border: { pt: 1, color: "C9D4E3" },
        fill: "FFFFFF",
        color: theme.text,
        margin: 0.05,
      }
    );
    addCard(
      s,
      7.05,
      3.95,
      5.65,
      2.4,
      "关键解释",
      "logistic_tabular 与 LLM Two-Stage 的 FP 规模几乎相当，\n" +
        "但前者每千笔可额外捕获约 30.2 笔违约。\n" +
        "当前差距核心在“违约识别能力”，而非“过度拒绝”。"
    );
    addFooter(s, 9);
  }

  // Slide 10: Discussion and limitations
  {
    const s = baseSlide(pptx);
    addTopBand(s, "9. 讨论、局限与下一步");
    addCard(
      s,
      0.75,
      1.2,
      6.0,
      5.6,
      "为什么结构化模型仍占优",
      "1) 高风险子集强化了结构化信号。\n" +
        "2) 两阶段 LLM 教师理由模板化，语义多样性有限。\n" +
        "3) Stage-2 决策头过于简化，难充分利用复杂上下文。\n" +
        "4) 模型规模与样本规模存在错配。\n" +
        "5) 无校准时易出现“高拒绝率换高召回”假优势。"
    );
    addCard(
      s,
      7.0,
      1.2,
      5.7,
      2.55,
      "有效性威胁",
      "• 主实验聚焦高风险子集，外推到全客群需谨慎。\n" +
        "• 当前缺少严格时间滚动验证。\n" +
        "• desc 字段历史可用性不均衡。"
    );
    addCard(
      s,
      7.0,
      3.95,
      5.7,
      2.85,
      "改进优先级",
      "高优先：教师信号升级 + LLM 概率校准。\n" +
        "中优先：结构化与文本联合判别头。\n" +
        "低优先：仅加复杂提示词（边际收益有限）。"
    );
    addFooter(s, 10);
  }

  // Slide 11: Reproducibility and takeaways
  {
    const s = baseSlide(pptx);
    addTopBand(s, "10. 可复现性与最终结论");
    addCard(
      s,
      0.7,
      1.1,
      6.15,
      3.05,
      "代码与结果映射",
      "workflow_common.py：清洗、切分、阈值、统一指标\n" +
        "ml_standalone/run_ml_pipeline.py：传统 ML\n" +
        "bert_standalone/run_bert_pipeline.py：BERT 路线\n" +
        "llm_standalone/run_llm_two_stage.py：两阶段 LLM\n" +
        "output/doc/report_tables/*.csv：论文主表来源"
    );
    addCard(
      s,
      0.7,
      4.35,
      6.15,
      2.55,
      "复现流程",
      "1) 重建共享切分并运行 ML/DA/BERT。\n" +
        "2) 运行 two-stage 与 one-stage LLM 脚本。\n" +
        "3) 更新外部 GPT 预测评估。\n" +
        "4) 生成论文与本演示文档。"
    );
    addCard(
      s,
      7.1,
      1.1,
      5.55,
      5.8,
      "最终结论",
      "• 在固定拒绝率公平约束下，结构化模型仍是主判首选。\n" +
        "• 两阶段 LLM 当前更适合作为解释与复核辅助层。\n" +
        "• 未来可行路线：结构化主判 + LLM 解释 + 人工复核联动。\n\n" +
        "Thank you.\nQ&A",
      {}
    );
    addFooter(s, 11);
  }

  fs.mkdirSync(path.dirname(outFile), { recursive: true });
  await pptx.writeFile({ fileName: outFile });
  process.stdout.write(`PPTX generated: ${outFile}\n`);
}

build().catch((err) => {
  console.error(err);
  process.exit(1);
});

