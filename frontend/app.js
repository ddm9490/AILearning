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
const libraryEmptyEl = document.getElementById("library-empty");
const libraryListEl = document.getElementById("library-list");
const libraryDetailPanelEl = document.getElementById("library-detail-panel");
const libraryDetailEl = document.getElementById("library-detail");

const state = {
  selectedKeywords: new Set(),
  chipAddWrap: null,
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

async function deleteJSON(url) {
  const res = await fetch(url, { method: "DELETE" });
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

// 사용자가 직접 입력한 키워드(예: "ConvNeXt")는 AI가 자연어 관심사를 검색어로
// "번역"하는 과정을 거치지 않고 arXiv 검색어에 그대로 들어간다(app.py 참고) — 특정
// 아키텍처/모델 이름을 AI가 엉뚱하게 해석해버리는 문제를 막기 위한 입력 경로.
function addExactKeyword(rawKeyword) {
  const keyword = rawKeyword.trim();
  if (!keyword || state.selectedKeywords.has(keyword)) return;

  state.selectedKeywords.add(keyword);

  const chip = document.createElement("button");
  chip.type = "button";
  chip.className = "chip active chip-custom";
  chip.textContent = `${keyword} ×`;
  chip.title = "클릭하면 제거돼요";
  chip.addEventListener("click", () => {
    state.selectedKeywords.delete(keyword);
    chip.remove();
  });

  // + 버튼(chipAddWrap)은 항상 칩 목록 맨 끝에 있어야 해서, 새 칩은 그 앞에 끼워 넣는다.
  if (state.chipAddWrap) {
    topicChipsEl.insertBefore(chip, state.chipAddWrap);
  } else {
    topicChipsEl.appendChild(chip);
  }
}

// 동그란 "+" 버튼 -> 클릭하면 인라인 입력창으로 바뀌는 태그 추가 컨트롤.
// Enter로 추가하고, 입력창은 열린 채로 유지해서 여러 개를 연달아 추가할 수 있다.
// 빈 채로 Escape/포커스 아웃하면 다시 + 버튼으로 접힌다.
function buildChipAddControl() {
  const wrap = document.createElement("span");
  wrap.className = "chip-add-wrap";

  const button = document.createElement("button");
  button.type = "button";
  button.className = "chip-add-btn";
  button.textContent = "+";
  button.setAttribute("aria-label", "키워드 직접 추가");

  const input = document.createElement("input");
  input.type = "text";
  input.className = "chip-add-input";
  input.placeholder = "키워드 입력 후 Enter";
  input.hidden = true;

  const collapse = () => {
    input.hidden = true;
    input.value = "";
    button.hidden = false;
  };

  const expand = () => {
    button.hidden = true;
    input.hidden = false;
    input.focus();
  };

  button.addEventListener("click", expand);

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      addExactKeyword(input.value);
      input.value = "";
    } else if (event.key === "Escape") {
      collapse();
    }
  });

  input.addEventListener("blur", () => {
    if (!input.value.trim()) collapse();
  });

  wrap.append(button, input);
  return wrap;
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
    badge.textContent = `#${kw}`;
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

  card.append(top, meta);

  // 선수 지식/추천 키워드를 나란히 2단으로 배치한다 — "필요한 것 vs 이 논문이
  // 다루는 것"을 바로 대비해서 볼 수 있고, 카드가 넓어져서(1080px 컨테이너)
  // 한 줄에 넉넉히 들어간다. 카드/화면이 좁아지면 grid-template-columns가
  // auto-fit이라 자동으로 다시 세로로 쌓인다.
  const keywordGroups = document.createElement("div");
  keywordGroups.className = "keyword-groups-row";
  if (paper.prerequisites && paper.prerequisites.length) {
    keywordGroups.appendChild(buildKeywordGroup("읽기 전 필요한 선수 지식", paper.prerequisites));
  }
  keywordGroups.appendChild(buildKeywordGroup("AI 추천 키워드", paper.keywords));
  card.appendChild(keywordGroups);

  // 요약문은 카드가 너무 길어 보이지 않게 기본 접어두고, 태그 바로 아래(버튼 줄
  // 위쪽) 자리에서 펼쳐진다. 키워드/선수 지식 태그는 한눈에 훑어보는 용도라 계속
  // 바로 보이게 둔다.
  const summary = document.createElement("p");
  summary.className = "paper-summary";
  summary.textContent = paper.summary;
  summary.hidden = true;
  card.appendChild(summary);

  const detailsBtn = document.createElement("button");
  detailsBtn.type = "button";
  detailsBtn.className = "curriculum-btn";
  detailsBtn.textContent = "자세히 보기";
  detailsBtn.addEventListener("click", () => {
    summary.hidden = !summary.hidden;
    detailsBtn.textContent = summary.hidden ? "자세히 보기" : "접기";
  });

  // "자세히 보기"와 "커리큘럼에 추가"를 같은 스타일 버튼으로 한 줄에 나란히 둔다.
  const { button: curriculumBtn, content: curriculumContent } = buildCurriculumTrigger(
    { target_type: "paper", title: paper.title, summary: paper.summary, pdf_url: paper.pdf_url },
    "커리큘럼에 추가",
  );

  const actionsRow = document.createElement("div");
  actionsRow.className = "curriculum-actions-row";
  actionsRow.append(detailsBtn, curriculumBtn);

  const section = document.createElement("div");
  section.className = "curriculum-section";
  section.append(actionsRow, curriculumContent);

  card.appendChild(section);

  return card;
}

function buildTagRow(keywords) {
  const row = document.createElement("div");
  row.className = "tag-row";

  const sorted = [...keywords].sort((a, b) => (a.tier ?? 99) - (b.tier ?? 99));
  sorted.forEach((kw) => {
    const tag = document.createElement("span");
    tag.className = kw.tier === null || kw.tier === undefined ? "tag tag-neutral" : `tag tier-${kw.tier}`;
    tag.textContent = `# ${kw.name}`;
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
// buttonLabel: 처음 누를 때(아직 생성 전) 보여줄 버튼 문구.
// 논문 카드에서 미리보기 삼아 눌러볼 때마다 "내 커리큘럼"에 자동으로 쌓이는 게
// 불편하다는 피드백을 받아서, 여기서는 save:false로 생성만 하고(미리보기), 마음에
// 들면 별도 "내 커리큘럼에 저장" 버튼으로 명시적으로 저장하게 분리했다.
// 감싸는 wrapper는 안 만들고 {button, content}만 돌려준다 — 호출부(buildPaperCard)가
// 이 버튼을 "자세히 보기" 버튼과 같은 줄에 나란히 놓고 싶어해서, 레이아웃은 호출부가
// 정하게 뺐다.
function buildCurriculumTrigger(target, buttonLabel) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "curriculum-btn";
  button.textContent = buttonLabel;

  const content = document.createElement("div");
  content.className = "curriculum-content";
  content.style.display = "none";

  let previewData = null;

  button.addEventListener("click", async () => {
    if (content.style.display !== "none") {
      content.style.display = "none";
      button.textContent = "커리큘럼 보기";
      return;
    }
    content.style.display = "block";
    if (previewData) {
      button.textContent = "커리큘럼 접기";
      return;
    }

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
        save: false,
      });
      previewData = data;
      // renderCurriculumGraph()가 맨 앞에서 container.innerHTML을 비우기 때문에,
      // 저장 바를 먼저 넣으면 그래프를 그리는 순간 같이 지워진다 — 그래프를 먼저
      // 그린 뒤 저장 바를 맨 위에 꽂아야 한다.
      renderCurriculumGraph(content, data);
      content.prepend(buildCurriculumPreviewSaveBar(data));
      button.textContent = "커리큘럼 접기";
    } catch (err) {
      const errorMsg = document.createElement("p");
      errorMsg.className = "curriculum-error";
      errorMsg.textContent = err.message || "커리큘럼을 만들지 못했어요.";
      content.appendChild(errorMsg);
      button.textContent = originalText;
    } finally {
      button.disabled = false;
    }
  });

  return { button, content };
}

// 미리보기 그래프 위에 뜨는 "내 커리큘럼에 저장" 바. data는 save:false로 받은 응답
// (id가 없는 상태) — 저장 전까지는 완료 표시/AI 설명/퀴즈 같은 id가 필요한 기능은
// 못 쓰고, 저장하고 나면 "내 커리큘럼" 탭에서 그 기능들을 이어서 쓸 수 있다.
function buildCurriculumPreviewSaveBar(data) {
  const bar = document.createElement("div");
  bar.className = "curriculum-preview-bar";

  const note = document.createElement("span");
  note.className = "curriculum-preview-note";
  note.textContent = "미리보기예요 — 저장해야 진행 상황이 남아요.";

  const saveBtn = document.createElement("button");
  saveBtn.type = "button";
  saveBtn.className = "curriculum-action-btn";
  saveBtn.textContent = "내 커리큘럼에 저장";

  saveBtn.addEventListener("click", async () => {
    saveBtn.disabled = true;
    saveBtn.textContent = "저장하는 중...";
    try {
      await postJSON("/api/curriculum/save", {
        target_label: data.target_label,
        target_type: data.target_type,
        used_rag: data.used_rag,
        nodes: data.nodes,
        edges: data.edges,
      });
      saveBtn.textContent = "저장됨";
      saveBtn.classList.add("curriculum-action-btn-done");
      note.textContent = "\"내 커리큘럼\" 탭에서 진행 상황을 이어갈 수 있어요.";
    } catch (err) {
      saveBtn.disabled = false;
      saveBtn.textContent = "내 커리큘럼에 저장";
      note.textContent = err.message || "저장하지 못했어요. 다시 시도해주세요.";
    }
  });

  bar.append(note, saveBtn);
  return bar;
}

const SVG_NS = "http://www.w3.org/2000/svg";
const GRAPH_NODE_WIDTH = 190;
const GRAPH_TARGET_NODE_WIDTH = 220;
const GRAPH_NODE_MIN_HEIGHT = 72;
const GRAPH_NODE_MAX_HEIGHT = 168;
const GRAPH_COL_GAP = 90;
const GRAPH_ROW_GAP = 22;
const GRAPH_PADDING = 24;

function createSvgEl(tag, attrs) {
  const el = document.createElementNS(SVG_NS, tag);
  Object.entries(attrs || {}).forEach(([key, value]) => el.setAttribute(key, value));
  return el;
}

function nodeBoxWidth(node) {
  return node.is_target ? GRAPH_TARGET_NODE_WIDTH : GRAPH_NODE_WIDTH;
}

// 실제 렌더링(foreignObject 안)과 높이 측정(measureNodeHeight)이 항상 같은 DOM
// 구조를 쓰게 하나로 합쳐뒀다 — 둘이 따로 놀면(예: 측정할 땐 타이틀만 재고 실제로는
// "목표" 라벨까지 얹는 식) 측정값이 실제 렌더링보다 작아져서 다시 텍스트가 잘린다.
function buildNodeBodyEl(node) {
  const body = document.createElement("div");
  body.className = "curriculum-node-body";
  if (node.is_target) {
    const eyebrow = document.createElement("span");
    eyebrow.className = "curriculum-node-eyebrow";
    eyebrow.textContent = "🎯 학습 목표";
    body.appendChild(eyebrow);
  }
  const title = document.createElement("span");
  title.className = "curriculum-node-title";
  title.textContent = node.title;
  body.appendChild(title);
  return body;
}

// 타이틀 길이에 따라 노드 높이가 다른데(짧은 한 줄 vs 긴 두세 줄), 고정 높이(예전
// 76px)를 쓰면 긴 타이틀(특히 타겟 노드)이 overflow:hidden에 잘려버렸다. 실제
// .curriculum-node-body와 동일한 클래스/폭으로 화면 밖에 숨겨서 렌더링한 뒤 실제
// 줄바꿈 높이를 재는 게, 글자 수만으로 줄 수를 추정하는 것보다 훨씬 정확하다
// (한글/영어가 섞여 있어 글자당 폭이 들쭉날쭉하기 때문).
let _measureHost = null;
function measureNodeHeight(node, width) {
  if (!_measureHost) {
    _measureHost = document.createElement("div");
    _measureHost.style.cssText = "position:fixed; left:-9999px; top:0; visibility:hidden; pointer-events:none;";
    document.body.appendChild(_measureHost);
  }
  const wrapper = document.createElement("div");
  if (node.is_target) wrapper.className = "curriculum-node-target";
  const body = buildNodeBodyEl(node);
  body.style.width = `${width}px`;
  body.style.height = "auto";
  wrapper.appendChild(body);
  _measureHost.appendChild(wrapper);
  const measured = body.scrollHeight;
  _measureHost.removeChild(wrapper);
  return Math.min(GRAPH_NODE_MAX_HEIGHT, Math.max(GRAPH_NODE_MIN_HEIGHT, measured));
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
  const colWidths = [];
  let maxColHeight = 0;

  layers.forEach((layer, colIndex) => {
    const colNodes = byLayer.get(layer);
    const colWidth = Math.max(...colNodes.map(nodeBoxWidth));
    colWidths.push(colWidth);
    const colX = GRAPH_PADDING + colWidths.slice(0, colIndex).reduce((sum, w) => sum + w + GRAPH_COL_GAP, 0);

    let y = GRAPH_PADDING;
    colNodes.forEach((node) => {
      const nodeWidth = nodeBoxWidth(node);
      const nodeHeight = measureNodeHeight(node, nodeWidth);
      positions.set(node.id, { x: colX, y, width: nodeWidth, height: nodeHeight });
      y += nodeHeight + GRAPH_ROW_GAP;
    });
    maxColHeight = Math.max(maxColHeight, y - GRAPH_ROW_GAP);
  });

  const totalColsWidth = colWidths.reduce((sum, w) => sum + w, 0) + Math.max(0, colWidths.length - 1) * GRAPH_COL_GAP;
  const width = GRAPH_PADDING * 2 + totalColsWidth;
  const height = GRAPH_PADDING * 2 + maxColHeight;
  return {
    positions,
    width: Math.max(width, GRAPH_NODE_WIDTH + GRAPH_PADDING * 2),
    height: Math.max(height, GRAPH_NODE_MIN_HEIGHT + GRAPH_PADDING * 2),
  };
}

function curriculumEdgePath(from, to) {
  const x1 = from.x + from.width;
  const y1 = from.y + from.height / 2;
  const x2 = to.x;
  const y2 = to.y + to.height / 2;
  const curve = Math.max(40, (x2 - x1) / 2);
  return `M ${x1} ${y1} C ${x1 + curve} ${y1}, ${x2 - curve} ${y2}, ${x2} ${y2}`;
}

function nodePrimaryTier(node) {
  const tiers = node.concepts.map((c) => c.tier).filter((t) => t !== null && t !== undefined);
  return tiers.length ? Math.min(...tiers) : null;
}

// 아직 완료 안 한 노드 중, 선수 노드(들어오는 간선의 from)가 전부 완료된 것들 —
// "지금 바로 공부해도 되는" 노드 집합. 그래프/완료 상태만으로 계산되는 순수 함수라
// LLM 호출이 전혀 없다(뼈대는 이미 DAG로 다 갖고 있으니 순회만 하면 됨).
function computeUnlockedNodeIds(nodes, edges) {
  const completedIds = new Set(nodes.filter((n) => n.completed).map((n) => n.id));
  const prereqsByNode = new Map();
  nodes.forEach((n) => prereqsByNode.set(n.id, []));
  (edges || []).forEach((e) => {
    if (prereqsByNode.has(e.to)) prereqsByNode.get(e.to).push(e.from);
  });

  const unlocked = new Set();
  nodes.forEach((n) => {
    if (completedIds.has(n.id)) return;
    const prereqs = prereqsByNode.get(n.id) || [];
    if (prereqs.every((p) => completedIds.has(p))) unlocked.add(n.id);
  });
  return unlocked;
}

// 완료된 노드 개수를 진행률 바 + 라벨로 보여준다. refresh()를 나중에도 다시 불러서
// 완료 토글이 일어날 때마다 갱신할 수 있게 엘리먼트 참조를 클로저에 들고 있는다.
function buildProgressSummary(nodes) {
  const wrap = document.createElement("div");
  wrap.className = "curriculum-progress";

  const bar = document.createElement("div");
  bar.className = "curriculum-progress-bar";
  const fill = document.createElement("div");
  fill.className = "curriculum-progress-fill";
  bar.appendChild(fill);

  const label = document.createElement("span");
  label.className = "curriculum-progress-label";

  const refresh = () => {
    const total = nodes.length;
    const completed = nodes.filter((n) => n.completed).length;
    const pct = total ? Math.round((completed / total) * 100) : 0;
    fill.style.width = `${pct}%`;
    label.textContent = `${completed} / ${total} 완료`;
    wrap.classList.toggle("curriculum-progress-complete", total > 0 && completed === total);
  };
  refresh();

  wrap.append(bar, label);
  return { el: wrap, refresh };
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

  const progress = buildProgressSummary(data.nodes);
  container.appendChild(progress.el);

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

  const nextUpEl = document.createElement("div");
  nextUpEl.className = "curriculum-nextup";
  const nodeGroups = new Map();

  const nodeContext = {
    curriculumId: data.id,
    refreshProgress: progress.refresh,
    refreshNextUp: () => refreshNextUp(data.nodes, data.edges || [], nodeGroups, nextUpEl, detailPanel, nodeContext),
  };

  const nodesGroup = createSvgEl("g", { class: "curriculum-nodes" });
  data.nodes.forEach((node) => {
    const pos = positions.get(node.id);
    if (!pos) return;

    const tier = nodePrimaryTier(node);
    const classes = ["curriculum-node"];
    if (node.is_target) classes.push("curriculum-node-target");
    if (tier !== null) classes.push(`tier-${tier}`);
    if (node.completed) classes.push("curriculum-node-completed");

    const g = createSvgEl("g", {
      class: classes.join(" "),
      transform: `translate(${pos.x}, ${pos.y})`,
      tabindex: "0",
    });

    g.appendChild(createSvgEl("rect", { width: pos.width, height: pos.height, rx: 12 }));

    // tier를 카드 배경 전체가 아니라 왼쪽 얇은 액센트 바 하나로만 표시한다 — 배경을
    // 통째로 칠하면(예전 방식) 노드마다 색이 서로 경쟁해서 그래프 전체가 산만해
    // 보였다. 카드는 전부 같은 중립 표면(rect fill)을 쓰고, tier 색은 이 바 하나에만
    // 실어서 "무슨 tier인지"는 여전히 한눈에 구분되지만 화면은 훨씬 차분해진다.
    g.appendChild(
      createSvgEl("rect", {
        x: 4,
        y: 8,
        width: 4,
        height: Math.max(0, pos.height - 16),
        rx: 2,
        class: "curriculum-node-accent",
      })
    );

    const foreignObject = createSvgEl("foreignObject", { width: pos.width, height: pos.height });
    foreignObject.appendChild(buildNodeBodyEl(node));
    g.appendChild(foreignObject);

    // 완료 체크 배지: 항상 DOM엔 있고, .curriculum-node-completed일 때만 CSS로 보여준다
    // (완료 토글 때마다 새로 만들지 않고 클래스만 바꾸면 되도록).
    const badge = createSvgEl("g", { class: "curriculum-node-badge" });
    badge.appendChild(createSvgEl("circle", { cx: pos.width - 12, cy: 12, r: 10 }));
    const check = createSvgEl("path", {
      d: `M ${pos.width - 17} 12 l 4 4 l 7 -8`,
      class: "curriculum-node-check",
    });
    badge.appendChild(check);
    g.appendChild(badge);

    const highlightEdges = (on) => {
      const selector = `[data-from="${CSS.escape(node.id)}"], [data-to="${CSS.escape(node.id)}"]`;
      edgesGroup.querySelectorAll(selector).forEach((p) => p.classList.toggle("curriculum-edge-active", on));
    };
    g.addEventListener("mouseenter", () => highlightEdges(true));
    g.addEventListener("mouseleave", () => highlightEdges(false));
    g.addEventListener("click", () => showCurriculumDetail(detailPanel, node, nodeContext, g));
    g.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        showCurriculumDetail(detailPanel, node, nodeContext, g);
      }
    });

    nodeGroups.set(node.id, g);
    nodesGroup.appendChild(g);
  });
  svg.appendChild(nodesGroup);

  container.appendChild(nextUpEl);
  nodeContext.refreshNextUp();

  graphWrap.appendChild(svg);
  container.append(graphWrap, detailPanel);
}

// "다음 학습 추천": 선수 노드를 전부 완료한, 아직 안 끝낸 노드들을 배너(클릭하면 상세
// 패널이 열림) + 그래프 안 하이라이트(.curriculum-node-next)로 동시에 보여준다.
// LLM 호출 없이 순수 그래프 계산이라 완료 토글마다 즉시 다시 불러도 비용이 없다.
function refreshNextUp(nodes, edges, nodeGroups, nextUpEl, detailPanel, context) {
  const unlocked = computeUnlockedNodeIds(nodes, edges);

  nodeGroups.forEach((g, nodeId) => {
    g.classList.toggle("curriculum-node-next", unlocked.has(nodeId));
  });

  nextUpEl.innerHTML = "";
  if (!unlocked.size) return;

  const label = document.createElement("span");
  label.className = "curriculum-nextup-label";
  label.textContent = "다음 학습 추천";
  nextUpEl.appendChild(label);

  nodes
    .filter((n) => unlocked.has(n.id))
    .forEach((n) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "curriculum-nextup-chip";
      chip.textContent = n.title;
      chip.addEventListener("click", () => showCurriculumDetail(detailPanel, n, context, nodeGroups.get(n.id)));
      nextUpEl.appendChild(chip);
    });
}

// panel: 설명을 그릴 컨테이너. node: 클릭된 노드 데이터(완료 상태를 여기 직접 mutate함).
// context: {curriculumId, refreshProgress}. nodeGroupEl: 이 노드의 SVG <g> (완료 시각
// 효과를 클래스 토글로 바로 반영하기 위함).
function showCurriculumDetail(panel, node, context, nodeGroupEl) {
  panel.innerHTML = "";

  const title = document.createElement("h4");
  title.textContent = node.is_target ? `${node.title} (최종 목표)` : node.title;
  panel.appendChild(title);

  const desc = document.createElement("p");
  desc.textContent = node.description;
  panel.appendChild(desc);

  if (node.learning_points && node.learning_points.length) {
    const label = document.createElement("div");
    label.className = "curriculum-detail-label";
    label.textContent = "이 단계에서 할 수 있어야 하는 것";
    panel.appendChild(label);

    const list = document.createElement("ul");
    list.className = "curriculum-learning-points";
    node.learning_points.forEach((point) => {
      const li = document.createElement("li");
      li.textContent = point;
      list.appendChild(li);
    });
    panel.appendChild(list);
  }

  panel.appendChild(buildTagRow(node.concepts));

  if (context && context.curriculumId) {
    const actions = document.createElement("div");
    actions.className = "curriculum-detail-actions";

    const explainSection = buildExplainSection(node, context);
    const quizSection = buildQuizSection(node, context, nodeGroupEl);

    // 버튼 3개는 항상 한 줄에 나란히, 펼쳐지는 내용(설명 텍스트/퀴즈 문제)은
    // 그 줄 아래에 따로 둔다 — 버튼과 내용을 같은 컨테이너에 넣으면 그 컨테이너
    // 자체가 한 덩어리로 줄바꿈돼서 버튼들이 나란히 안 보이는 문제가 있었다.
    actions.append(
      buildCompleteToggleButton(node, context, nodeGroupEl),
      explainSection.button,
      quizSection.button,
    );
    panel.append(actions, explainSection.content, quizSection.content);
  }
}

function buildCompleteToggleButton(node, context, nodeGroupEl) {
  const button = document.createElement("button");
  button.type = "button";
  // curriculum-action-btn-complete: 아직 완료 전일 때의 연한 초록 배경. 완료하면
  // setLabel()이 curriculum-action-btn-done(진한 초록)을 같이 붙여서 덮어쓴다.
  button.className = "curriculum-action-btn curriculum-action-btn-complete";

  const setLabel = () => {
    button.textContent = node.completed ? "✓ 완료함 (취소하려면 클릭)" : "학습 완료로 표시";
    button.classList.toggle("curriculum-action-btn-done", node.completed);
  };
  setLabel();

  button.addEventListener("click", async () => {
    const nextState = !node.completed;
    button.disabled = true;
    try {
      await postJSON(`/api/curriculum/${context.curriculumId}/nodes/${node.id}/complete`, {
        completed: nextState,
      });
      node.completed = nextState;
      setLabel();
      if (nodeGroupEl) nodeGroupEl.classList.toggle("curriculum-node-completed", nextState);
      if (context.refreshProgress) context.refreshProgress();
      if (context.refreshNextUp) context.refreshNextUp();
    } catch (err) {
      console.error(err);
    } finally {
      button.disabled = false;
    }
  });

  return button;
}

// {button, content}만 돌려준다(감싸는 wrap 없이) — 완료/설명/퀴즈 버튼 3개를
// showCurriculumDetail이 한 줄에 나란히 놓고, 펼쳐지는 내용(설명 텍스트)은 그
// 버튼 줄과 별도로 아래쪽에 배치하기 위해서다. 버튼 하나와 그 결과물을 같은
// div로 묶어두면(예전처럼) 그 div 자체가 flex row 안에서 한 덩어리로 취급돼서
// 옆의 다른 버튼들과 나란히 안 놓이고 줄바꿈되는 문제가 있었다.
function buildExplainSection(node, context) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "curriculum-action-btn curriculum-action-btn-explain";
  button.textContent = "AI에게 더 자세히 설명 요청";

  const textEl = document.createElement("p");
  textEl.className = "curriculum-explain-text";

  if (node.ai_explanation) {
    textEl.textContent = node.ai_explanation;
    button.hidden = true;
  } else {
    textEl.hidden = true;
  }

  button.addEventListener("click", async () => {
    button.disabled = true;
    const originalText = button.textContent;
    button.textContent = "AI가 설명을 만드는 중... (최대 1분 정도 걸려요)";
    try {
      const data = await postJSON(`/api/curriculum/${context.curriculumId}/nodes/${node.id}/explain`, {});
      node.ai_explanation = data.explanation;
      textEl.textContent = data.explanation;
      textEl.hidden = false;
      button.hidden = true;
    } catch (err) {
      textEl.textContent = err.message || "설명을 가져오지 못했어요.";
      textEl.hidden = false;
      button.disabled = false;
      button.textContent = originalText;
    }
  });

  return { button, content: textEl };
}

// "학습 완료" 버튼은 자기 신고제라 실제 이해를 검증할 방법이 없었다 — 이 퀴즈가 그
// 검증 루프. 객관식이라 채점은 서버 호출 없이 클라이언트에서 바로 되고, 문제 자체는
// (explain처럼) 한 번 생성되면 캐싱되어 다시 눌러도 API를 또 부르지 않는다.
// explain과 마찬가지로 {button, content}만 돌려준다 — 버튼 줄과 퀴즈 본문을
// showCurriculumDetail이 따로 배치한다.
function buildQuizSection(node, context, nodeGroupEl) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "curriculum-action-btn curriculum-action-btn-quiz";
  button.textContent = "이해도 확인 퀴즈 풀기";

  const quizEl = document.createElement("div");
  quizEl.className = "curriculum-quiz-wrap";
  quizEl.hidden = true;

  const showQuiz = (questions) => {
    button.hidden = true;
    quizEl.hidden = false;
    renderQuiz(quizEl, questions, node, context, nodeGroupEl);
  };

  if (node.quiz) {
    showQuiz(node.quiz);
  }

  button.addEventListener("click", async () => {
    button.disabled = true;
    const originalText = button.textContent;
    button.textContent = "AI가 퀴즈를 만드는 중... (최대 1분 정도 걸려요)";
    try {
      const data = await postJSON(`/api/curriculum/${context.curriculumId}/nodes/${node.id}/quiz`, {});
      node.quiz = data.questions;
      showQuiz(data.questions);
    } catch (err) {
      button.disabled = false;
      button.textContent = originalText;
      alert(err.message || "퀴즈를 만들지 못했어요.");
    }
  });

  return { button, content: quizEl };
}

function renderQuiz(container, questions, node, context, nodeGroupEl) {
  container.innerHTML = "";

  const form = document.createElement("div");
  form.className = "curriculum-quiz";

  const questionEls = questions.map((q, qIndex) => {
    const qWrap = document.createElement("div");
    qWrap.className = "curriculum-quiz-question";

    const qText = document.createElement("p");
    qText.className = "curriculum-quiz-question-text";
    qText.textContent = `${qIndex + 1}. ${q.question}`;
    qWrap.appendChild(qText);

    const optionsWrap = document.createElement("div");
    optionsWrap.className = "curriculum-quiz-options";
    q.options.forEach((optionText, optIndex) => {
      const optionLabel = document.createElement("label");
      optionLabel.className = "curriculum-quiz-option";

      const radio = document.createElement("input");
      radio.type = "radio";
      radio.name = `quiz-${node.id}-q${qIndex}`;
      radio.value = String(optIndex);

      optionLabel.append(radio, document.createTextNode(optionText));
      optionsWrap.appendChild(optionLabel);
    });
    qWrap.appendChild(optionsWrap);

    const feedback = document.createElement("p");
    feedback.className = "curriculum-quiz-feedback";
    feedback.hidden = true;
    qWrap.appendChild(feedback);

    form.appendChild(qWrap);
    return { qWrap, optionsWrap, feedback };
  });

  const submitBtn = document.createElement("button");
  submitBtn.type = "button";
  submitBtn.className = "curriculum-action-btn";
  submitBtn.textContent = "채점하기";

  const resultEl = document.createElement("p");
  resultEl.className = "curriculum-quiz-result";
  resultEl.hidden = true;

  submitBtn.addEventListener("click", () => {
    let correctCount = 0;
    questions.forEach((q, qIndex) => {
      const { optionsWrap, feedback } = questionEls[qIndex];
      const selected = optionsWrap.querySelector("input:checked");
      const selectedIndex = selected ? Number(selected.value) : -1;
      const isCorrect = selectedIndex === q.correct_index;
      if (isCorrect) correctCount += 1;

      feedback.hidden = false;
      feedback.textContent = isCorrect
        ? `정답이에요! ${q.explanation}`
        : `정답은 "${q.options[q.correct_index]}"예요. ${q.explanation}`;
      feedback.classList.toggle("curriculum-quiz-feedback-correct", isCorrect);
      feedback.classList.toggle("curriculum-quiz-feedback-wrong", !isCorrect);

      optionsWrap.querySelectorAll("input").forEach((input) => (input.disabled = true));
    });

    submitBtn.hidden = true;
    resultEl.hidden = false;
    resultEl.textContent = `${questions.length}문제 중 ${correctCount}개 맞혔어요.`;

    if (!node.completed) {
      const completeBtn = document.createElement("button");
      completeBtn.type = "button";
      completeBtn.className = "curriculum-action-btn curriculum-action-btn-done";
      completeBtn.textContent = "학습 완료로 표시";
      completeBtn.addEventListener("click", async () => {
        completeBtn.disabled = true;
        try {
          await postJSON(`/api/curriculum/${context.curriculumId}/nodes/${node.id}/complete`, {
            completed: true,
          });
          node.completed = true;
          if (nodeGroupEl) nodeGroupEl.classList.add("curriculum-node-completed");
          if (context.refreshProgress) context.refreshProgress();
          if (context.refreshNextUp) context.refreshNextUp();
          completeBtn.textContent = "✓ 완료 처리됨";
        } catch (err) {
          completeBtn.disabled = false;
          alert(err.message || "완료 처리에 실패했어요.");
        }
      });
      resultEl.after(completeBtn);
    }
  });

  container.append(form, submitBtn, resultEl);
}

const TARGET_TYPE_ICON = { paper: "📄", keyword: "🧭" };

function formatCreatedAt(unixSeconds) {
  return new Date(unixSeconds * 1000).toLocaleString("ko-KR", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

async function renderLibraryList() {
  libraryListEl.innerHTML = "";
  libraryDetailPanelEl.hidden = true;

  let items;
  try {
    items = await fetchJSON("/api/curricula");
  } catch (err) {
    libraryEmptyEl.textContent = "커리큘럼 목록을 불러오지 못했어요.";
    libraryEmptyEl.style.display = "block";
    return;
  }

  if (!items.length) {
    libraryEmptyEl.style.display = "block";
    return;
  }
  libraryEmptyEl.style.display = "none";

  items.forEach((item) => {
    libraryListEl.appendChild(buildLibraryRow(item));
  });
}

function buildLibraryRow(item) {
  const row = document.createElement("div");
  row.className = "library-row";

  const main = document.createElement("button");
  main.type = "button";
  main.className = "library-row-main";

  const titleLine = document.createElement("div");
  titleLine.className = "library-row-title";
  titleLine.textContent = `${TARGET_TYPE_ICON[item.target_type] || "🧭"} ${item.target_label}`;

  const metaLine = document.createElement("div");
  metaLine.className = "library-row-meta";
  const pct = item.total_nodes ? Math.round((item.completed_nodes / item.total_nodes) * 100) : 0;
  metaLine.textContent = `${formatCreatedAt(item.created_at)} · ${item.completed_nodes} / ${item.total_nodes} 완료`;

  const miniBar = document.createElement("div");
  miniBar.className = "library-row-bar";
  const miniFill = document.createElement("div");
  miniFill.className = "library-row-bar-fill";
  miniFill.style.width = `${pct}%`;
  miniBar.appendChild(miniFill);

  main.append(titleLine, metaLine, miniBar);
  main.addEventListener("click", () => openLibraryCurriculum(item.id));

  const deleteBtn = document.createElement("button");
  deleteBtn.type = "button";
  deleteBtn.className = "library-row-delete";
  deleteBtn.textContent = "삭제";
  deleteBtn.title = "이 커리큘럼 삭제";
  deleteBtn.addEventListener("click", async (event) => {
    event.stopPropagation();
    if (!confirm(`"${item.target_label}" 커리큘럼을 삭제할까요?`)) return;
    try {
      await deleteJSON(`/api/curriculum/${item.id}`);
      renderLibraryList();
    } catch (err) {
      alert(err.message || "삭제하지 못했어요.");
    }
  });

  row.append(main, deleteBtn);
  return row;
}

async function openLibraryCurriculum(curriculumId) {
  libraryDetailEl.innerHTML = "불러오는 중...";
  libraryDetailPanelEl.hidden = false;
  libraryDetailPanelEl.scrollIntoView({ behavior: "smooth", block: "start" });

  try {
    const data = await fetchJSON(`/api/curriculum/${curriculumId}`);
    renderCurriculumGraph(libraryDetailEl, data);
  } catch (err) {
    libraryDetailEl.textContent = err.message || "커리큘럼을 불러오지 못했어요.";
  }
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
      if (target === "library") renderLibraryList();
    });
  });
}

async function init() {
  searchForm.addEventListener("submit", handleSubmit);
  keywordCurriculumForm.addEventListener("submit", handleKeywordCurriculumSubmit);
  initTabs();

  renderSuggestedKeywords();
  state.chipAddWrap = buildChipAddControl();
  topicChipsEl.appendChild(state.chipAddWrap);

  const tiers = await fetchJSON("/api/tiers");
  renderTierLegend(tiers);
}

init();
