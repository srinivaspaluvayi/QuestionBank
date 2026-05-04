import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import express from "express";
import multer from "multer";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA    = path.join(__dirname, "Data");
const Q_DIR   = path.join(DATA, "questions");
const S_DIR   = path.join(DATA, "solutions");
const IMG_DIR = path.join(DATA, "images");
const JSON_FILE = path.join(DATA, "questions.json");

// Make sure all folders exist
fs.mkdirSync(Q_DIR,   { recursive: true });
fs.mkdirSync(S_DIR,   { recursive: true });
fs.mkdirSync(IMG_DIR, { recursive: true });
if (!fs.existsSync(JSON_FILE)) fs.writeFileSync(JSON_FILE, "[]");

function readJSON() {
  return JSON.parse(fs.readFileSync(JSON_FILE, "utf8"));
}
function writeJSON(data) {
  fs.writeFileSync(JSON_FILE, JSON.stringify(data, null, 2));
}
function nextId(entries) {
  const ids = entries.map(e => parseInt(e.id)).filter(n => !isNaN(n));
  return String(ids.length ? Math.max(...ids) + 1 : 1);
}

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 500 * 1024 * 1024, fieldSize: 500 * 1024 * 1024 }
}).any();

const app = express();
app.use("/static",         express.static(path.join(__dirname, "static")));
app.use("/media/questions",express.static(Q_DIR));
app.use("/media/solutions", express.static(S_DIR));
app.use("/media/images",   express.static(IMG_DIR));

app.get("/", (_req, res) =>
  res.sendFile(path.join(__dirname, "static", "index.html"))
);

// ── GET all entries (hydrated with file contents for display) ─────────────────
app.get("/api/entries", (_req, res) => {
  const entries = readJSON();
  const hydrated = entries.slice(-10).map(e => {
    const qFile = path.join(DATA, e.question);
    const sFile = path.join(DATA, e.solution);
    return {
      ...e,
      questionText: fs.existsSync(qFile) ? fs.readFileSync(qFile, "utf8") : "",
      solutionText: fs.existsSync(sFile) ? fs.readFileSync(sFile, "utf8") : "",
    };
  });
  res.json(hydrated);
});

// ── POST new entry ────────────────────────────────────────────────────────────
app.post("/api/entries", (req, res) => {
  upload(req, res, err => {
    if (err) {
      console.error("multer error:", err);
      return res.status(400).json({ error: err.message });
    }
    try {
      const body = req.body || {};
      const topic     = body.topic     || "";
      const subtopic  = body.subtopic  || "";
      const reference = body.reference || "";
      const question  = body.question  || "";
      const solution  = body.solution  || "";

      const entries = readJSON();
      const id = nextId(entries);

      fs.writeFileSync(path.join(Q_DIR, `${id}.txt`), question);
      fs.writeFileSync(path.join(S_DIR, `${id}.txt`), solution);

      const images = [];
      for (const f of (req.files || []).filter(f => f.fieldname === "images")) {
        const ext  = path.extname(f.originalname || ".bin").toLowerCase() || ".bin";
        const name = `${id}_${images.length + 1}${ext}`;
        fs.writeFileSync(path.join(IMG_DIR, name), f.buffer);
        images.push(`./images/${name}`);
      }

      const record = {
        id,
        question:  `./questions/${id}.txt`,
        solution:  `./solutions/${id}.txt`,
        images,
        topic,
        subtopic,
        reference,
      };
      entries.push(record);
      writeJSON(entries);

      res.json({ ...record, questionText: question, solutionText: solution });
    } catch (e) {
      console.error("POST /api/entries error:", e);
      res.status(500).json({ error: e.message });
    }
  });
});

// ── DELETE entry ──────────────────────────────────────────────────────────────
app.delete("/api/entries/:id", (req, res) => {
  const { id } = req.params;
  let entries = readJSON();
  const entry = entries.find(e => e.id === id);
  if (!entry) return res.status(404).json({ error: "Not found" });

  entries = entries.filter(e => e.id !== id);
  writeJSON(entries);

  const tryDel = p => { try { fs.unlinkSync(p); } catch {} };
  tryDel(path.join(DATA, entry.question || `./questions/${id}.txt`));
  tryDel(path.join(DATA, entry.solution || `./solutions/${id}.txt`));
  (entry.images || []).forEach(img => {
    // support both "./images/file.png" (new) and bare "file.png" (legacy)
    const p = img.includes("/") ? path.join(DATA, img) : path.join(IMG_DIR, img);
    tryDel(p);
  });

  res.json({ ok: true });
});

const PORT = process.env.PORT || 8765;
app.listen(PORT, () => console.log(`http://127.0.0.1:${PORT}`));
