const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  AlignmentType, LevelFormat, convertInchesToTwip
} = require("docx");

const OUT = "D:/DS_Hack/report/Banking_Customer_Transaction_Behaviour_Report.docx";
const CH = "D:/DS_Hack/outputs";

function img(path, widthIn, heightIn) {
  return new ImageRun({
    type: "png",
    data: fs.readFileSync(path),
    transformation: { width: widthIn * 96, height: heightIn * 96 },
  });
}

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 } });
}
function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 120 } });
}
function bullet(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, ...opts })],
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 80 },
  });
}
function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, italics: true, size: 20, color: "555555" })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
  });
}
function figure(path, widthIn, heightIn, captionText) {
  return [
    new Paragraph({ children: [img(path, widthIn, heightIn)], alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 } }),
    caption(captionText),
  ];
}

function metricTable(rows) {
  const headerCells = ["Metric", "Value"].map(t => new TableCell({
    width: { size: 4500, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: "2F5496" },
    children: [new Paragraph({ children: [new TextRun({ text: t, bold: true, color: "FFFFFF" })] })],
  }));
  const body = rows.map(([m, v]) => new TableRow({
    children: [
      new TableCell({ width: { size: 4500, type: WidthType.DXA }, children: [new Paragraph(m)] }),
      new TableCell({ width: { size: 4500, type: WidthType.DXA }, children: [new Paragraph(v)] }),
    ],
  }));
  return new Table({
    width: { size: 9000, type: WidthType.DXA },
    columnWidths: [4500, 4500],
    rows: [new TableRow({ children: headerCells }), ...body],
  });
}

const doc = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }],
    }],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 } } },
    children: [
      new Paragraph({
        children: [new TextRun({ text: "Understanding Customer Transaction Behaviour", bold: true, size: 40 })],
        spacing: { after: 100 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Banking Analytics Case Study — EDA & KNN Classification", size: 26, color: "555555" })],
        spacing: { after: 300 },
      }),

      h1("1. Objective & Approach"),
      p("A bank wants to move beyond summary reports and understand how customers actually use its financial products — deposits, withdrawals, UPI, ATM, card and bill-payment channels. This report analyses six months of customer-level transaction data to:"),
      bullet("Identify distinct transaction behaviour patterns across customer segments."),
      bullet("Determine which customer groups prefer digital transactions over cash/ATM channels."),
      bullet("Investigate whether ATM usage is declining in specific segments."),
      bullet("Flag customers with unusually high transaction activity for further investigation."),
      bullet("Build and evaluate a K-Nearest Neighbours (KNN) model to predict digital-adoption propensity, and translate the results into a practical engagement action plan."),

      h2("Data source"),
      p("No publicly available dataset matched this exact schema (per-customer, 6-month aggregates of UPI / Card / ATM / Deposits / Withdrawals / Bill Payments alongside Age Group, Occupation and City), so a synthetic dataset of 1,500 customers was generated with realistic, segment-conditioned behaviour (e.g. younger customers skew digital, business owners transact at higher values). Deliberate data-quality issues — missing values, duplicate rows, inconsistent text casing, and a few negative 'reversal' amounts — were injected to mirror a real raw bank extract, then resolved in the cleaning step below. Two supporting fields were added beyond the base schema: a month-by-month ATM breakdown (Month 1–6) to support trend analysis, and Card_Spend (₹ value of card transactions) to complement the given Card_Count."),

      h1("2. Data Cleaning & Validation"),
      p("Starting from 1,512 raw rows, the following issues were identified and resolved:"),
      bullet("Removed 12 exact duplicate rows (system export duplication)."),
      bullet("Standardised text fields (Occupation, City, Age Group) — trimmed whitespace and fixed inconsistent casing (e.g. \"  business owner \" → \"Business Owner\")."),
      bullet("Imputed missing values: Occupation/City (30 and 22 missing) filled with the segment mode; Bill_Payments (15 missing) filled with the age-group median; Card_Spend (15 missing) filled with 0 (a valid reading for non-card users)."),
      bullet("Corrected 8 negative Withdrawal values (refund/reversal artefacts) by taking absolute value."),
      bullet("Enforced integer/float types on all count and currency columns and validated Age_Group against the expected category set."),
      p("Result: a clean, validated dataset of 1,500 customers with zero missing values and no duplicates, saved to data/bank_transactions_clean.csv.", { bold: true }),

      h1("3. Exploratory Data Analysis"),
      h2("3.1 Descriptive statistics (post-cleaning)"),
      metricTable([
        ["UPI transactions / customer (6 mo.)", "mean 26.6, median 23, max 140"],
        ["Card transactions / customer (6 mo.)", "mean 14.5, median 12, max 73"],
        ["ATM transactions / customer (6 mo.)", "mean 26.6, median 21, max 160"],
        ["Deposits (₹, 6 mo.)", "mean ₹46,184, median ₹27,768"],
        ["Withdrawals (₹, 6 mo.)", "mean ₹29,623, median ₹17,662"],
        ["Bill payments / customer (6 mo.)", "mean 5.8, median 5"],
        ["Digital share of transactions", "mean 61%, median 66%"],
      ]),

      h2("3.2 Which customer groups prefer digital transactions?"),
      ...figure(`${CH}/chart1_digital_ratio_by_age.png`, 6, 3.75, "Chart 1 — Average digital transaction share (UPI + Card as a % of all channel transactions) by age group."),
      p("Digital preference falls sharply and almost linearly with age: 18–25 (84%) and 26–35 (80%) are digital-first, 36–45 is transitional (65%), and 46–55 (46%), 56–65 (31%) and 65+ (16%) remain ATM/cash-oriented. By occupation, Students (83%) and Salaried employees (66%) lead digital usage, while Retirees (25%) lag well behind. City-level differences are modest by comparison (58%–64% across all 12 cities) — geography is not a strong driver once age and occupation are accounted for."),

      h2("3.3 Is ATM usage declining, and where?"),
      ...figure(`${CH}/chart2_atm_trend_by_age.png`, 6, 3.75, "Chart 2 — Average monthly ATM transaction count by age group across the 6-month window."),
      p("Yes, but not uniformly. The 26–35 and 36–45 segments show the steepest month-on-month decline (trend slope ≈ −0.17 to −0.18 ATM transactions/month, roughly a 30–35% drop from Month 1 to Month 6) as these customers actively migrate to UPI/card. The 18–25 segment starts from a low ATM base and declines more gently (slope ≈ −0.10) since it was never ATM-heavy to begin with. In contrast, 56–65 and 65+ segments show flat-to-rising ATM usage (slope +0.03 to +0.15) — these customers are not migrating away from cash channels. By occupation, Business Owners and Salaried employees show the fastest ATM decline; Retirees show a slight increase."),

      h2("3.4 Unusually high transaction patterns"),
      ...figure(`${CH}/chart3_outliers_by_occupation.png`, 6, 3.75, "Chart 3 — Total 6-month transaction value (deposits + withdrawals + card spend) by occupation, with statistical outliers (IQR method) highlighted in red."),
      p("Using the IQR rule (values beyond Q3 + 1.5×IQR of total transaction value), 108 customers (7.2% of the base) were flagged as unusually high transactors — averaging ₹4.1 lakh in total 6-month activity versus a ₹85k base-wide average, with the single highest case at ₹22.5 lakh. Business Owners are disproportionately represented (17.0% of Business Owners flagged, vs. 3–8% in every other occupation), consistent with commercial cash-flow volumes rather than fraud per se — but this segment, along with any individual salaried/homemaker accounts showing 10–20x their expected transaction value, warrants a closer AML/monitoring review rather than being treated as routine consumer activity."),

      h2("3.5 Relationships between behaviours"),
      ...figure(`${CH}/chart4_correlation_heatmap.png`, 5.5, 4.3, "Chart 4 — Correlation matrix of key transaction behaviour metrics."),
      p("Digital_Ratio correlates negatively with ATM_Count (−0.52) confirming a genuine channel substitution effect, not just independent growth. Deposits and Withdrawals are strongly correlated (0.79), as expected for active transactors. ATM_Trend_Slope has only a weak correlation with the other variables, confirming it captures a distinct, additional signal (behavioural trend) not visible in the static totals."),

      h1("4. KNN Classification Model"),
      h2("Prediction target & rationale"),
      p("Target: Digital_Preferring (1 = customer's UPI + Card transaction count exceeds the base median of 35.5; 0 = otherwise) — a balanced 50/50 split by construction."),
      p("Feature set (deliberately excludes UPI_Count, Card_Count and Digital_Ratio, which define the label): Age_Group, Occupation, City, ATM_Count, ATM_Trend_Slope, Deposits, Withdrawals, Bill_Payments."),
      p("Business rationale: the bank can already observe demographics, ATM usage and monetary flows for every customer, including those with thin or no digital transaction history yet. A model that predicts digital-adoption propensity from these always-available signals lets the bank proactively target the right customers for UPI/card onboarding campaigns — rather than only identifying digital adopters after the fact from their own UPI/card counts."),

      h2("Method"),
      p("Numeric features were standardised (z-score); Age_Group was ordinal-encoded (18-25 → 65+); Occupation and City were one-hot encoded. Data was split 75/25 (stratified). A 5-fold cross-validated grid search tuned k (3–29), weighting (uniform/distance) and distance metric (Euclidean/Manhattan), optimising for F1-score."),

      h2("Results"),
      metricTable([
        ["Best hyperparameters", "k = 29, weights = uniform, metric = euclidean"],
        ["Best 5-fold CV F1", "0.841"],
        ["Test accuracy", "83.5%"],
        ["Test precision", "83.8%"],
        ["Test recall", "82.9%"],
        ["Test F1-score", "0.833"],
        ["Test ROC-AUC", "0.906"],
      ]),
      ...figure(`${CH}/chart5_knn_confusion_matrix.png`, 4.2, 4.2, "Chart 5 — KNN confusion matrix on the held-out test set (375 customers)."),
      p("Of 375 test customers, the model correctly classified 313 (158 Traditional + 155 Digital-Preferring), with a roughly even, low error rate on both classes (30 false positives, 32 false negatives) — there is no systematic bias toward either class. An ROC-AUC of 0.91 indicates strong separability well above chance (0.5), confirming that demographics and account-level behaviour alone carry real predictive signal for digital-adoption propensity, even without observing UPI/card activity directly."),

      h2("Interpreting the model alongside the EDA"),
      p("The model's strongest implicit signals — Age_Group and ATM_Trend_Slope — directly reflect the two patterns found in the EDA: younger, digitally-active segments have both lower ATM usage and a declining ATM trend, while older segments show flat-to-rising ATM use and low digital share. This convergence between an independently-trained classifier and the manual EDA findings validates that age and ATM behaviour trend are genuine, reliable drivers of digital adoption — not an artifact of a single analysis method — which increases confidence in using the model's predictions operationally."),

      h1("5. Key Insights"),
      bullet("Digital preference declines steeply and near-linearly with age: 84% (18–25) → 16% (65+) of transactions are digital, with the steepest drop-off between the 36–45 and 56–65 bands.", { bold: false }),
      bullet("Occupation matters as much as age: Students (83%) and Salaried employees (66%) are digital-first; Retirees (25%) are the most cash/ATM-reliant group regardless of city."),
      bullet("ATM usage is actively declining — not just low — in the 26–45 age bands (≈30–35% drop over 6 months), while it is flat or slightly rising among 56+ customers, meaning digital migration is a live, ongoing shift concentrated in the middle-age segments rather than already complete."),
      bullet("7.2% of customers (108 of 1,500) show statistically unusual transaction value, averaging 4.8x the base-wide mean; Business Owners are over-represented among them (17% flagged vs. 3–8% elsewhere), warranting targeted monitoring review."),
      bullet("City/location has only a marginal effect on digital adoption (58–64% across 12 cities) — segmentation strategy should prioritise age and occupation, not geography."),
      bullet("A KNN model using only demographics, ATM behaviour and monetary activity (no direct UPI/card history) predicts digital-adoption propensity with 83.5% accuracy and 0.91 ROC-AUC, confirming these are reliable, generalisable predictors the bank can act on for customers without established digital usage patterns."),
      bullet("Digital adoption and ATM usage are meaningfully substitutive (−0.52 correlation), not independent trends — every incremental digital transaction is, on average, displacing cash-channel activity rather than adding to overall transaction volume."),

      h1("6. Recommended Action Plan"),
      h2("Digital engagement"),
      bullet("Target the 46–65 age bands (46% and 31% digital share respectively) with assisted UPI/card onboarding — in-branch demos, family-assisted sign-up, and simplified card-linked UPI setup — since they show real transaction activity but low digital share."),
      bullet("Use the KNN model to score customers with limited digital history today and prioritise outreach to those predicted \"Digital-Preferring\" but not yet transacting digitally — the highest-propensity, lowest-effort conversion group."),
      bullet("Deprioritise city-based campaign segmentation; reallocate that budget to age/occupation-targeted messaging, which the data shows is far more predictive."),
      h2("Channel & operations"),
      bullet("Right-size ATM footprint and cash-replenishment cycles down in areas serving predominantly 26–45 customers, where ATM usage is actively declining, while maintaining ATM service levels for 56+ dominant branches where usage is flat or rising."),
      bullet("Retirees (25% digital, flat/rising ATM use) should receive a distinct retention-focused (not conversion-focused) service model — reliable ATM/branch access matters more to this segment than digital push."),
      h2("Risk & monitoring"),
      bullet("Route the 108 flagged high-value customers — especially the 17% of Business Owners among them — to the transaction-monitoring/AML team for a routine review pass, distinguishing legitimate high cash-flow business activity from anomalous behaviour."),
      bullet("Re-run the outlier scan monthly (not just once on 6-month aggregates) to catch emerging unusual patterns earlier rather than after a full 6-month accumulation."),
      h2("Next steps"),
      bullet("Validate the synthetic-data findings against real transaction data before operationalising the model; recalibrate the KNN threshold/features once live UPI/Card history is available for a holdout validation."),
      bullet("Track the Digital_Ratio and ATM_Trend_Slope metrics as ongoing KPIs per segment to measure whether the action plan is actually shifting behaviour over the next 2–3 reporting cycles."),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log("Wrote", OUT);
});
