const POLL_INTERVAL_MS = 2000;
let currentRun = null;
let lastRenderedLength = 0;

async function fetchJson(url) {
    const r = await fetch(url);
    return r.json();
}

async function fetchText(url) {
    const r = await fetch(url);
    if (!r.ok) return null;
    return r.text();
}

async function loadRuns() {
    const runs = await fetchJson("/api/runs");
    const select = document.getElementById("run-selector");
    select.innerHTML = "";
    for (const r of runs) {
        const opt = document.createElement("option");
        opt.value = `${r.exp_id}/${r.timestamp}`;
        opt.textContent = `${r.exp_id} @ ${r.timestamp}`;
        select.appendChild(opt);
    }
    if (runs.length > 0) {
        currentRun = `${runs[0].exp_id}/${runs[0].timestamp}`;
        select.value = currentRun;
    }
    select.addEventListener("change", () => {
        currentRun = select.value;
        lastRenderedLength = 0;
        document.getElementById("script").innerHTML = "";
        refresh();
    });
}

function renderImpressions(listElem, impressions) {
    listElem.innerHTML = "";
    if (!impressions || impressions.length === 0) {
        const li = document.createElement("li");
        li.className = "empty";
        li.textContent = "(空帳本 — 本場未帶記憶入場)";
        listElem.appendChild(li);
        return;
    }
    for (const imp of impressions) {
        const li = document.createElement("li");
        const text = document.createElement("div");
        text.className = "imp-text";
        text.textContent = imp.text;
        const meta = document.createElement("div");
        meta.className = "imp-meta";
        const fromRun = imp.from_run ? imp.from_run.split("/").pop() : "—";
        meta.textContent = `Turn ${imp.from_turn} · score ${imp.score} · ${fromRun}`;
        li.appendChild(text);
        li.appendChild(meta);
        listElem.appendChild(li);
    }
}

async function refreshRetrieval() {
    if (!currentRun) return;
    const data = await fetchJson(`/api/retrieval?run=${encodeURIComponent(currentRun)}`);
    if (!data) return;
    const protagonist = data.protagonist || {};
    const counterpart = data.counterpart || {};
    document.getElementById("role-left-name").textContent = protagonist.persona_name || "母親";
    document.getElementById("role-right-name").textContent = counterpart.persona_name || "兒子";
    renderImpressions(document.getElementById("impressions-left"), protagonist.impressions);
    renderImpressions(document.getElementById("impressions-right"), counterpart.impressions);
}

async function refreshConversation() {
    if (!currentRun) return;
    const text = await fetchText(`/api/conversation?run=${encodeURIComponent(currentRun)}`);
    if (text === null) return;
    if (text.length === lastRenderedLength) return;
    const scriptEl = document.getElementById("script");
    const isFirstRender = lastRenderedLength === 0;
    scriptEl.innerHTML = marked.parse(text);
    // Scroll to bottom on new content
    scriptEl.scrollTop = scriptEl.scrollHeight;
    if (isFirstRender) {
        scriptEl.classList.remove("fade-in");
        void scriptEl.offsetWidth;  // reflow
        scriptEl.classList.add("fade-in");
    } else {
        scriptEl.classList.remove("pulse");
        void scriptEl.offsetWidth;
        scriptEl.classList.add("pulse");
    }
    lastRenderedLength = text.length;
}

async function refreshMeta() {
    if (!currentRun) return;
    const data = await fetchJson(`/api/meta?run=${encodeURIComponent(currentRun)}`);
    const el = document.getElementById("meta-summary");
    if (!data) {
        el.textContent = "in progress...";
        return;
    }
    const tin = data.total_tokens_in || 0;
    const tout = data.total_tokens_out || 0;
    const tokens = `${tin.toLocaleString()}/${tout.toLocaleString()} tok`;
    const dur = data.duration_seconds ? `· ${data.duration_seconds.toFixed(1)}s` : "";
    const reason = data.termination_reason ? `· ${data.termination_reason}` : "";
    el.textContent = `Turn ${data.total_turns} · ${tokens} ${dur} ${reason}`;
}

async function refresh() {
    await Promise.all([
        refreshRetrieval(),
        refreshConversation(),
        refreshMeta(),
    ]);
}

async function main() {
    await loadRuns();
    await refresh();
    setInterval(refresh, POLL_INTERVAL_MS);
}

main();
