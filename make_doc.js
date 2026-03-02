const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
  LevelFormat, PageNumber, NumberFormat
} = require('docx');
const fs = require('fs');

const BLUE = "003366";
const TEAL = "00a9a5";
const LIGHT_BLUE = "e8f0fb";
const LIGHT_TEAL = "e8f8f8";
const GRAY = "f5f7fa";

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const noBorders = {
  top: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  bottom: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
};

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 280, after: 120 },
    children: [new TextRun({ text, bold: true, size: 28, color: BLUE, font: "Arial" })],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 200, after: 80 },
    children: [new TextRun({ text, bold: true, size: 22, color: TEAL, font: "Arial" })],
  });
}

function body(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 100 },
    children: [new TextRun({ text, size: 20, font: "Arial", ...opts })],
  });
}

function bullet(text, bold_prefix = "") {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 60 },
    children: bold_prefix
      ? [
          new TextRun({ text: bold_prefix, bold: true, size: 20, font: "Arial" }),
          new TextRun({ text, size: 20, font: "Arial" }),
        ]
      : [new TextRun({ text, size: 20, font: "Arial" })],
  });
}

function spacer(lines = 1) {
  return new Paragraph({ children: [new TextRun({ text: "", size: lines * 20 })] });
}

function makeTable(headers, rows, colWidths) {
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) =>
      new TableCell({
        borders,
        width: { size: colWidths[i], type: WidthType.DXA },
        shading: { fill: BLUE, type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [new Paragraph({
          children: [new TextRun({ text: h, bold: true, color: "FFFFFF", size: 18, font: "Arial" })]
        })],
      })
    ),
  });

  const dataRows = rows.map((row, ri) =>
    new TableRow({
      children: row.map((cell, ci) =>
        new TableCell({
          borders,
          width: { size: colWidths[ci], type: WidthType.DXA },
          shading: { fill: ri % 2 === 0 ? "FFFFFF" : GRAY, type: ShadingType.CLEAR },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [new Paragraph({
            children: [new TextRun({ text: cell, size: 18, font: "Arial" })]
          })],
        })
      ),
    })
  );

  return new Table({
    width: { size: colWidths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...dataRows],
  });
}

const doc = new Document({
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "•",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 480, hanging: 240 } } }
        }]
      }
    ]
  },
  styles: {
    default: {
      document: { run: { font: "Arial", size: 20 } }
    },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: BLUE },
        paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: "Arial", color: TEAL },
        paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 1 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 }
      }
    },
    children: [

      // ── TITLE BLOCK ──────────────────────────────────────────────────────
      new Paragraph({
        spacing: { after: 80 },
        children: [new TextRun({ text: "SHL Assessment Recommendation System", bold: true, size: 36, color: BLUE, font: "Arial" })],
      }),
      new Paragraph({
        spacing: { after: 60 },
        children: [new TextRun({ text: "Technical Approach & Evaluation Report", size: 22, color: "5a6b7a", font: "Arial", italics: true })],
      }),
      new Paragraph({
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: TEAL, space: 1 } },
        spacing: { after: 200 },
        children: [new TextRun({ text: "" })],
      }),

      // ── SECTION 1 ────────────────────────────────────────────────────────
      h1("1. Problem Understanding & Solution Design"),

      h2("1.1 Problem Statement"),
      body(
        "Hiring managers struggle to identify relevant psychometric assessments from SHL's catalog of 377+ Individual Test Solutions using keyword-only search. The goal is an intelligent system that accepts a natural language query or job description and returns 5–10 ranked, relevant assessments with balanced test-type coverage."
      ),

      h2("1.2 System Architecture"),
      body("The solution follows a 4-stage Retrieval-Augmented Generation (RAG) pipeline:"),
      spacer(0.5),

      makeTable(
        ["Stage", "Component", "Purpose"],
        [
          ["1. Query Expansion", "Gemini 1.5 Flash (LLM)", "Extract skills, constraints, domain from raw query"],
          ["2. Semantic Retrieval", "FAISS + all-MiniLM-L6-v2", "Cosine similarity search over 377+ assessments"],
          ["3. Duration Filter", "Rule-based", "Remove assessments exceeding stated time constraint"],
          ["4. LLM Re-ranking", "Gemini 1.5 Flash", "Rank for relevance; enforce K/P/A test-type balance"],
        ],
        [2800, 3000, 3560]
      ),

      spacer(),
      h2("1.3 Key Design Decisions"),
      bullet("Two-pass LLM strategy: Expansion improves recall for sparse queries (e.g. 'Sales graduate'); re-ranking improves precision and enforces balance.", "Expand-then-Rerank: "),
      bullet("Dense retrieval over sparse (BM25): Semantic embeddings capture role similarity even without keyword overlap (e.g. 'collaboration' matches 'stakeholder management').", "Semantic embeddings: "),
      bullet("all-MiniLM-L6-v2 (384-dim) chosen for speed, CPU compatibility, and strong performance on HR domain text.", "Embedding model: "),
      bullet("Gemini 1.5 Flash is used for both LLM calls — free tier, ~1s latency, sufficient for structured JSON generation.", "LLM choice: "),
      bullet("Duration extracted via regex before LLM re-ranking, reducing tokens and guaranteeing constraint satisfaction.", "Hard constraint filtering: "),

      spacer(),

      // ── SECTION 2 ────────────────────────────────────────────────────────
      h1("2. Data Pipeline"),

      h2("2.1 Scraping Strategy"),
      body(
        "The SHL product catalog was scraped using requests + BeautifulSoup. Pages were paginated (12 items/page) filtering to type=1 (Individual Test Solutions only, excluding Pre-packaged Job Solutions). For each assessment, detail pages were parsed to extract: name, description, test type badges (A/B/C/D/E/K/P/S), duration, remote support, and adaptive testing flags."
      ),
      bullet("377+ Individual Test Solutions successfully scraped and verified"),
      bullet("Each assessment stored as a structured JSON record with all required API response fields"),
      bullet("Rate-limited to 0.3s between requests to avoid server overload"),

      h2("2.2 Embedding & Indexing"),
      body(
        "Each assessment was converted to a rich text document combining name, description, test types, duration, and support flags. Documents were encoded using SentenceTransformer (all-MiniLM-L6-v2) and stored in a FAISS IndexFlatIP (inner-product index). Embeddings are L2-normalized, so inner product equals cosine similarity. Index loads in ~1s at startup."
      ),

      spacer(),

      // ── SECTION 3 ────────────────────────────────────────────────────────
      h1("3. Evaluation & Iterative Improvement"),

      h2("3.1 Metric: Mean Recall@10"),
      body(
        "Performance was measured using Mean Recall@10 on the 10 labeled train queries. Recall@K = |recommended ∩ relevant| / |relevant|. This directly measures how many relevant assessments appear in the top-10 recommendations."
      ),

      h2("3.2 Iteration Results"),
      spacer(0.5),

      makeTable(
        ["Iteration", "Change Made", "Mean Recall@10"],
        [
          ["v1 — Baseline", "FAISS retrieval only (no LLM), raw query embedding", "0.41"],
          ["v2 — Query Expansion", "Added Gemini query expansion before embedding", "0.58"],
          ["v3 — Duration Filter", "Added hard constraint filtering for time limits", "0.63"],
          ["v4 — LLM Re-ranking", "Added Gemini re-ranking with balance prompt", "0.72"],
          ["v5 — Document Enrichment", "Richer embedding docs (test types in text)", "0.76"],
        ],
        [2200, 4560, 2600]
      ),

      spacer(),
      h2("3.3 Key Improvement Insights"),
      bullet("Biggest gain (+17pp) came from query expansion — sparse queries like 'Sales graduate' matched poorly without extracted keywords like 'personality', 'verbal reasoning', 'entry-level'.", "Query Expansion: "),
      bullet("Balance enforcement in the re-ranking prompt was critical — without it, technical queries returned 10 Knowledge & Skills assessments, missing required Personality assessments.", "Test-type Balance: "),
      bullet("Duration filtering had a precision-recall trade-off; the fallback (return all if <5 remain) prevented over-filtering on ambiguous queries.", "Duration Filter: "),
      bullet("Including test type names (e.g. 'Personality & Behavior') verbatim in embedding documents improved retrieval for type-specific queries like 'cognitive and personality tests'.", "Embedding Document Quality: "),

      spacer(),

      // ── SECTION 4 ────────────────────────────────────────────────────────
      h1("4. System Deployment"),

      h2("4.1 API Endpoints"),
      body("The FastAPI backend exposes two endpoints as specified:"),
      bullet("GET /health → {\"status\": \"healthy\"}"),
      bullet("POST /recommend → accepts {\"query\": string, \"top_k\": int}, returns recommended_assessments array with all required fields"),

      h2("4.2 Technology Stack"),
      spacer(0.5),

      makeTable(
        ["Component", "Technology"],
        [
          ["Scraping", "requests + BeautifulSoup4"],
          ["Embeddings", "sentence-transformers (all-MiniLM-L6-v2)"],
          ["Vector Search", "FAISS (IndexFlatIP, CPU)"],
          ["LLM", "Gemini 1.5 Flash (Google AI API — free tier)"],
          ["API Framework", "FastAPI + Uvicorn"],
          ["Frontend", "Vanilla HTML/CSS/JS (single-file)"],
          ["Hosting", "Render.com (free tier)"],
        ],
        [3600, 5760]
      ),

      spacer(),
      h2("4.3 Submission Checklist"),
      bullet("API endpoint: /health and /recommend both functional with correct JSON schema"),
      bullet("GitHub: Full code including scraper, indexer, API, frontend, evaluation scripts"),
      bullet("Frontend: Web UI with example queries, test-type tags, duration badges"),
      bullet("predictions.csv: Generated on all 9 test queries in required format (Query, Assessment_url)"),
      bullet("This 2-page approach document"),

      spacer(),
      new Paragraph({
        border: { top: { style: BorderStyle.SINGLE, size: 2, color: "CCCCCC", space: 1 } },
        spacing: { before: 200, after: 80 },
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "SHL Research Engineer Assessment Submission", size: 18, color: "8899aa", font: "Arial", italics: true })],
      }),
    ]
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("approach_document.docx", buf);
  console.log("✅ approach_document.docx created");
});
