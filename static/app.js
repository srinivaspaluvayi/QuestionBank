const form = document.getElementById("form");
const msg  = document.getElementById("msg");
const list = document.getElementById("list");

function esc(s) {
  return String(s ?? "")
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

async function load() {
  const res = await fetch("/api/entries");
  const entries = await res.json();
  if (!entries.length) {
    list.innerHTML = "<p class='empty'>No entries yet.</p>";
    return;
  }
  entries.slice().reverse().forEach(e => {
    const div = document.createElement("div");
    div.className = "entry";
    const imgs = (e.images || []).map(imgPath => {
      const name = imgPath.replace(/^\.\/images\//, "");
      return `<img src="/media/images/${encodeURIComponent(name)}" />`;
    }).join("");

    div.innerHTML = `
      <div class="entry-top">
        <strong>#${esc(e.id)} &mdash; ${esc(e.topic)}</strong>
        <button class="del" onclick="del('${esc(e.id)}')">Delete</button>
      </div>
      ${e.subtopic ? `<div class="meta">Subtopic: ${esc(e.subtopic)}</div>` : ""}
      ${e.reference ? `<div class="meta">Ref: ${esc(e.reference)}</div>` : ""}
      <details><summary>Question</summary><pre>${esc(e.questionText)}</pre></details>
      <details><summary>Solution</summary><pre>${esc(e.solutionText)}</pre></details>
      ${imgs ? `<div class="thumbs">${imgs}</div>` : ""}
    `;
    list.appendChild(div);
  });
}

async function del(id) {
  if (!confirm(`Delete entry #${id}?`)) return;
  await fetch(`/api/entries/${id}`, { method: "DELETE" });
  list.innerHTML = "";
  load();
}

form.addEventListener("submit", async e => {
  e.preventDefault();
  msg.textContent = "Saving…";
  const fd = new FormData(form);
  const files = document.getElementById("imgs").files;
  fd.delete("images");
  for (const f of files) fd.append("images", f);

  const res  = await fetch("/api/entries", { method: "POST", body: fd });
  const body = await res.json();
  if (!res.ok) { msg.textContent = "Error: " + (body.error || res.statusText); return; }
  msg.textContent = `Saved #${body.id}`;
  form.reset();
  list.innerHTML = "";
  load();
});

load();

// ── Resizable splitter ────────────────────────────────────────────────────────
(function () {
  const splitter  = document.getElementById("splitter");
  const formPanel = document.getElementById("panel-form");
  const layout    = document.querySelector(".layout");
  let dragging = false, startX = 0, startW = 0;

  splitter.addEventListener("mousedown", e => {
    dragging = true;
    startX = e.clientX;
    startW = formPanel.offsetWidth;
    splitter.classList.add("dragging");
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  });

  document.addEventListener("mousemove", e => {
    if (!dragging) return;
    const dx = e.clientX - startX;
    const total = layout.offsetWidth;
    const newW = Math.min(Math.max(startW + dx, 200), total - 250);
    formPanel.style.width = newW + "px";
  });

  document.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    splitter.classList.remove("dragging");
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  });
})();
