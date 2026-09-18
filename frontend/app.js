const SUGGESTED_KEYWORDS = [
  "transformer",
  "reinforcement learning",
  "diffusion model",
  "graph neural network",
  "large language model",
  "self-supervised learning",
  "multimodal",
];

const interestInput = document.getElementById("interest-input");
const topicChipsEl = document.getElementById("topic-chips");
const countInput = document.getElementById("count-input");
const searchForm = document.getElementById("search-form");
const submitBtn = document.getElementById("submit-btn");
const resultsListEl = document.getElementById("results-list");
const resultsEmptyEl = document.getElementById("results-empty");
const resultsCountEl = document.getElementById("results-count");
const tierLegendEl = document.getElementById("tier-legend");
const searchKeywordsBannerEl = document.getElementById("search-keywords-banner");
const keywordCurriculumForm = document.getElementById("keyword-curriculum-form");
const keywordCurriculumInput = document.getElementById("keyword-curriculum-input");
const keywordCurriculumRagCheckbox = document.getElementById("keyword-curriculum-rag");
const keywordCurriculumSubmitBtn = document.getElementById("keyword-curriculum-submit-btn");
const keywordCurriculumResultEl = document.getElementById("keyword-curriculum-result");

const state = {
  selectedKeywords: new Set(),
};

async function postJSON(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(body.error || `요청 실패: ${res.status}`);
  }
  return body;
}

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`요청 실패: ${res.status}`);
  return res.json();
}

function renderSuggestedKeywords() {
  topicChipsEl.innerHTML = "";
  SUGGESTED_KEYWORDS.forEach((keyword) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = keyword;
    chip.addEventListener("click", () => toggleKeyword(keyword, chip));
    topicChipsEl.appendChild(chip);
  });
}

function toggleKeyword(keyword, chipEl) {
  if (state.selectedKeywords.has(keyword)) {
    state.selectedKeywords.delete(keyword);
    chipEl.classList.remove("active");
  } else {
    state.selectedKeywords.add(keyword);
    chipEl.classList.add("active");
  }
}

function renderTierLegend(tiers) {
  tierLegendEl.innerHTML = "";
  tiers.forEach((tier) => {
    const item = document.createElement("span");
    item.className = "legend-item";
    item.title = tier.description;

    const dot = document.createElement("span");
    dot.className = "legend-dot";
    dot.style.background = `var(--tier-${tier.id}-text)`;

    item.append(dot, document.createTextNode(tier.name));
    tierLegendEl.appendChild(item);
  });
}

function renderSearchKeywordsBanner(keywords) {
  if (!keywords || !keywords.length) {
    searchKeywordsBannerEl.style.display = "none";
    return;
  }
  searchKeywordsBannerEl.style.display = "flex";
  searchKeywordsBannerEl.innerHTML = "";

  const label = document.createElement("span");
  label.className = "search-keywords-label";
  label.textContent = "AI가 이 키워드로 검색했어요:";
  searchKeywordsBannerEl.appendChild(label);

  keywords.forEach((kw) => {
    const badge = document.createElement("span");
    badge.className = "search-keyword-badge";
    badge.textContent = kw;
    searchKeywordsBannerEl.appendChild(badge);
  });
}

function renderResults(papers) {
  resultsListEl.innerHTML = "";

  if (!papers.length) {
    resultsEmptyEl.textContent = "조건에 맞는 논문을 찾지 못했어요. 다른 관심사로 시도해보세요.";
    resultsEmptyEl.style.display = "block";
    resultsCountEl.textContent = "";
    return;
  }

  resultsEmptyEl.style.display = "none";
  resultsCountEl.textContent = `${papers.length}건`;

  papers.forEach((paper) => {
    resultsListEl.appendChild(buildPaperCard(paper));
  });
}

function formatAuthors(authors) {
  if (authors.length <= 3) return authors.join(", ");
  return `${authors.slice(0, 3).join(", ")} 외 ${authors.length - 3}명`;
}

function buildPaperCard(paper) {
  const card = document.createElement("article");
  card.className = "paper-card";

  const top = document.createElement("div");
  top.className = "paper-card-top";

  const title = document.createElement("h3");
  title.className = "paper-title";
  title.textContent = paper.title;

  const links = document.createElement("div");
  links.className = "paper-links";
  const absLink = document.createElement("a");
  absLink.href = paper.abs_url;
  absLink.target = "_blank";
  absLink.rel = "noopener noreferrer";
  absLink.textContent = "arXiv";
  const pdfLink = document.createElement("a");
  pdfLink.href = paper.pdf_url;
  pdfLink.target = "_blank";
  pdfLink.rel = "noopener noreferrer";
  pdfLink.textContent = "PDF";
  links.append(absLink, pdfLink);

  top.append(title, links);

  const meta = document.createElement("p");
  meta.className = "paper-meta";
  meta.textContent = `${paper.year} · ${formatAuthors(paper.authors)} · ${paper.categories.join(", ")}`;

  const summary = document.createElement("p");
  summary.className = "paper-summary";
  summary.textContent = paper.summary;

  card.append(top, meta, summary);
  if (paper.prerequisites && paper.prerequisites.length) {
    card.appendChild(buildKeywordGroup("읽기 전 필요한 선수 지식", paper.prerequisites));
  }
  card.appendChild(buildKeywordGroup("AI 추천 키워드", paper.keywords));
  card.appendChild(buildCurriculumSection({ target_type: "paper", title: paper.title, summary: paper.summary, pdf_url: paper.pdf_url }, "이 논문 커리큘럼 만들기"));

  return card;
}

function buildTagRow(keywords) {
  const row = document.createElement("div");
  row.className = "tag-row";

  const sorted = [...keywords].sort((a, b) => (a.tier ?? 99) - (b.tier ?? 99));
  sorted.forEach((kw) => {
    const tag = document.createElement("span");
    tag.className = kw.tier === null || kw.tier === undefined ? "tag tag-neutral" : `tag tier-${kw.tier}`;
    tag.textContent = kw.name;
    tag.title = kw.tier_name ?? "";
    row.appendChild(tag);
  });
  return row;
}

function buildKeywordGroup(label, keywords) {
  const group = document.createElement("div");
  group.className = "keyword-group";

  const labelEl = document.createElement("div");
  labelEl.className = "keyword-group-label";
  labelEl.textContent = label;

  group.append(labelEl, buildTagRow(keywords));
  return group;
}

// target: {target_type: "paper", title, summary, pdf_url} 또는 {target_type: "keyword", keyword}
// buttonLabel: 접었다 펼 때 보여줄 기본 버튼 문구
function buildCurriculumSection(target, buttonLabel) {
  const wrapper = document.createElement("div");
  wrapper.className = "curriculum-section";

  const button = document.createElement("button");
  button.type = "button";
  button.className = "curriculum-btn";
  button.textContent = buttonLabel;

  const content = document.createElement("div");
  content.className = "curriculum-content";
  content.style.display = "none";

  let loaded = false;

  button.addEventListener("click", async () => {
    if (content.style.display !== "none") {
      content.style.display = "none";
      return;
    }
    content.style.display = "block";
    if (loaded) return;

    button.disabled = true;
    const originalText = button.textContent;
    button.textContent = target.target_type === "paper"
      ? "AI가 PDF를 읽는 중... (최대 1~2분 걸려요)"
      : "AI가 관련 자료를 찾는 중...";
    content.innerHTML = "";

    try {
      const data = await postJSON("/api/curriculum", {
        ...target,
        interest: interestInput.value.trim(),
      });
      renderCurriculumGraph(content, data);
      loaded = true;
    } catch (err) {
      const errorMsg = document.createElement("p");
      errorMsg.className = "curriculum-error";
      errorMsg.textContent = err.message || "커리큘럼을 만들지 못했어요.";
      content.appendChild(errorMsg);
    } finally {
      button.disabled = false;
      button.textContent = originalText;
    }
  });

  wrapper.append(button, content);
  return wrapper;
}

const SVG_NS = "http://www.w3.org/2000/svg";
const GRAPH_NODE_WIDTH = 190;
const GRAPH_NODE_HEIGHT = 76;
const GRAPH_COL_GAP = 90;
const GRAPH_ROW_GAP = 22;
const GRAPH_PADDING = 24;

function createSvgEl(tag, attrs) {
  const el = document.createElementNS(SVG_NS, tag);
  Object.entries(attrs || {}).forEach(([key, value]) => el.setAttribute(key, value));
  return el;
}

function layoutCurriculumGraph(nodes) {
  const byLayer = new Map();
  nodes.forEach((node) => {
    const layer = node.layer ?? 0;
    if (!byLayer.has(layer)) byLayer.set(layer, []);
    byLayer.get(layer).push(node);
  });

  const layers = [...byLayer.keys()].sort((a, b) => a - b);
  const positions = new Map();
  let maxRows = 0;

  layers.forEach((layer, colIndex) => {
    const colNodes = byLayer.get(layer);
    maxRows = Math.max(maxRows, colNodes.length);
    colNodes.forEach((node, rowIndex) => {
      positions.set(node.id, {
        x: GRAPH_PADDING + colIndex * (GRAPH_NODE_WIDTH + GRAPH_COL_GAP),
        y: GRAPH_PADDING + rowIndex * (GRAPH_NODE_HEIGHT + GRAPH_ROW_GAP),
      });
    });
  });

  const width = GRAPH_PADDING * 2 + layers.length * GRAPH_NODE_WIDTH + Math.max(0, layers.length - 1) * GRAPH_COL_GAP;
  const height = GRAPH_PADDING * 2 + maxRows * GRAPH_NODE_HEIGHT + Math.max(0, maxRows - 1) * GRAPH_ROW_GAP;
  return { positions, width: Math.max(width, GRAPH_NODE_WIDTH + GRAPH_PADDING * 2), height: Math.max(height, GRAPH_NODE_HEIGHT + GRAPH_PADDING * 2) };
}

function curriculumEdgePath(from, to) {
  const x1 = from.x + GRAPH_NODE_WIDTH;
  const y1 = from.y + GRAPH_NODE_HEIGHT / 2;
  const x2 = to.x;
  const y2 = to.y + GRAPH_NODE_HEIGHT / 2;
  const curve = Math.max(40, (x2 - x1) / 2);
  return `M ${x1} ${y1} C ${x1 + curve} ${y1}, ${x2 - curve} ${y2}, ${x2} ${y2}`;
}

function nodePrimaryTier(node) {
  const tiers = node.concepts.map((c) => c.tier).filter((t) => t !== null && t !== undefined);
  return tiers.length ? Math.min(...tiers) : null;
}

function renderCurriculumGraph(container, data) {
  container.innerHTML = "";

  if (!data.nodes || !data.nodes.length) {
    const empty = document.createElement("p");
    empty.className = "curriculum-error";
    empty.textContent = "커리큘럼을 만들지 못했어요.";
    container.appendChild(empty);
    return;
  }

  const banner = document.createElement("p");
  banner.className = "curriculum-rag-banner";
  banner.textContent = data.used_rag
    ? "관련 자료를 찾아 근거로 삼아 만들었어요 (RAG)."
    : "AI의 사전 지식만으로 만들었어요 (RAG 미사용).";
  container.appendChild(banner);

  const { positions, width, height } = layoutCurriculumGraph(data.nodes);

  const graphWrap = document.createElement("div");
  graphWrap.className = "curriculum-graph-wrap";

  const svg = createSvgEl("svg", { width, height, viewBox: `0 0 ${width} ${height}` });

  const defs = createSvgEl("defs");
  const marker = createSvgEl("marker", {
    id: `curriculum-arrow-${Math.random().toString(36).slice(2, 8)}`,
    viewBox: "0 0 10 10",
    refX: "9",
    refY: "5",
    markerWidth: "7",
    markerHeight: "7",
    orient: "auto-start-reverse",
  });
  marker.appendChild(createSvgEl("path", { d: "M 0 0 L 10 5 L 0 10 z", class: "curriculum-arrow-head" }));
  defs.appendChild(marker);
  svg.appendChild(defs);
  const markerId = marker.getAttribute("id");

  const edgesGroup = createSvgEl("g", { class: "curriculum-edges" });
  (data.edges || []).forEach((edge) => {
    const from = positions.get(edge.from);
    const to = positions.get(edge.to);
    if (!from || !to) return;
    edgesGroup.appendChild(
      createSvgEl("path", {
        d: curriculumEdgePath(from, to),
        class: "curriculum-edge",
        "marker-end": `url(#${markerId})`,
        "data-from": edge.from,
        "data-to": edge.to,
      })
    );
  });
  svg.appendChild(edgesGroup);

  const detailPanel = document.createElement("div");
  detailPanel.className = "curriculum-detail";
  detailPanel.textContent = "노드를 클릭하면 자세한 설명이 여기 나와요.";

  const nodesGroup = createSvgEl("g", { class: "curriculum-nodes" });
  data.nodes.forEach((node) => {
    const pos = positions.get(node.id);
    if (!pos) return;

    const tier = nodePrimaryTier(node);
    const classes = ["curriculum-node"];
    if (node.is_target) classes.push("curriculum-node-target");
    if (tier !== null) classes.push(`tier-${tier}`);

    const g = createSvgEl("g", {
      class: classes.join(" "),
      transform: `translate(${pos.x}, ${pos.y})`,
      tabindex: "0",
    });

    g.appendChild(createSvgEl("rect", { width: GRAPH_NODE_WIDTH, height: GRAPH_NODE_HEIGHT, rx: 12 }));

    const foreignObject = createSvgEl("foreignObject", { width: GRAPH_NODE_WIDTH, height: GRAPH_NODE_HEIGHT });
    const body = document.createElement("div");
    body.className = "curriculum-node-body";
    body.textContent = node.title;
    foreignObject.appendChild(body);
    g.appendChild(foreignObject);

    const highlightEdges = (on) => {
      const selector = `[data-from="${CSS.escape(node.id)}"], [data-to="${CSS.escape(node.id)}"]`;
      edgesGroup.querySelectorAll(selector).forEach((p) => p.classList.toggle("curriculum-edge-active", on));
    };
    g.addEventListener("mouseenter", () => highlightEdges(true));
    g.addEventListener("mouseleave", () => highlightEdges(false));
    g.addEventListener("click", () => showCurriculumDetail(detailPanel, node));
    g.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        showCurriculumDetail(detailPanel, node);
      }
    });

    nodesGroup.appendChild(g);
  });
  svg.appendChild(nodesGroup);

  graphWrap.appendChild(svg);
  container.append(graphWrap, detailPanel);
}

function showCurriculumDetail(panel, node) {
  panel.innerHTML = "";

  const title = document.createElement("h4");
  title.textContent = node.is_target ? `${node.title} (최종 목표)` : node.title;

  const desc = document.createElement("p");
  desc.textContent = node.description;

  panel.append(title, desc, buildTagRow(node.concepts));
}

function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  submitBtn.textContent = isLoading ? "AI가 찾는 중... (최대 1분 정도 걸려요)" : "AI 추천받기";
}

async function handleSubmit(event) {
  event.preventDefault();
  const interest = interestInput.value.trim();
  const keywords = Array.from(state.selectedKeywords);
  const count = countInput.value;

  if (!interest && !keywords.length) {
    resultsEmptyEl.textContent = "관심 분야를 입력하거나 키워드를 선택해주세요.";
    resultsEmptyEl.style.display = "block";
    return;
  }

  setLoading(true);
  searchKeywordsBannerEl.style.display = "none";

  try {
    const data = await postJSON("/api/recommend", { interest, keywords, count });
    renderSearchKeywordsBanner(data.search_keywords);
    renderResults(data.papers);
  } catch (err) {
    resultsEmptyEl.textContent = err.message || "추천을 불러오지 못했어요.";
    resultsEmptyEl.style.display = "block";
    resultsListEl.innerHTML = "";
    resultsCountEl.textContent = "";
    console.error(err);
  } finally {
    setLoading(false);
  }
}

async function handleKeywordCurriculumSubmit(event) {
  event.preventDefault();
  const keyword = keywordCurriculumInput.value.trim();

  if (!keyword) {
    keywordCurriculumResultEl.innerHTML = "";
    const msg = document.createElement("p");
    msg.className = "curriculum-error";
    msg.textContent = "키워드를 입력해주세요.";
    keywordCurriculumResultEl.appendChild(msg);
    return;
  }

  keywordCurriculumSubmitBtn.disabled = true;
  const originalText = keywordCurriculumSubmitBtn.textContent;
  keywordCurriculumSubmitBtn.textContent = "AI가 커리큘럼을 만드는 중...";
  keywordCurriculumResultEl.innerHTML = "";

  try {
    const data = await postJSON("/api/curriculum", {
      target_type: "keyword",
      keyword,
      use_rag: keywordCurriculumRagCheckbox.checked,
      interest: interestInput.value.trim(),
    });
    renderCurriculumGraph(keywordCurriculumResultEl, data);
  } catch (err) {
    const msg = document.createElement("p");
    msg.className = "curriculum-error";
    msg.textContent = err.message || "커리큘럼을 만들지 못했어요.";
    keywordCurriculumResultEl.appendChild(msg);
  } finally {
    keywordCurriculumSubmitBtn.disabled = false;
    keywordCurriculumSubmitBtn.textContent = originalText;
  }
}

function initTabs() {
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanels = document.querySelectorAll(".tab-panel");

  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;
      tabButtons.forEach((b) => {
        const active = b === btn;
        b.classList.toggle("active", active);
        b.setAttribute("aria-selected", String(active));
      });
      tabPanels.forEach((panel) => {
        panel.hidden = panel.id !== `tab-panel-${target}`;
      });
    });
  });
}

async function init() {
  searchForm.addEventListener("submit", handleSubmit);
  keywordCurriculumForm.addEventListener("submit", handleKeywordCurriculumSubmit);
  initTabs();

  renderSuggestedKeywords();

  const tiers = await fetchJSON("/api/tiers");
  renderTierLegend(tiers);
}

init();
