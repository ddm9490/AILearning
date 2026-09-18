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
  card.appendChild(buildKeywordGroup("AI 추천 키워드", paper.keywords));

  return card;
}

function buildKeywordGroup(label, keywords) {
  const group = document.createElement("div");
  group.className = "keyword-group";

  const labelEl = document.createElement("div");
  labelEl.className = "keyword-group-label";
  labelEl.textContent = label;

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

  group.append(labelEl, row);
  return group;
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

async function init() {
  searchForm.addEventListener("submit", handleSubmit);

  renderSuggestedKeywords();

  const tiers = await fetchJSON("/api/tiers");
  renderTierLegend(tiers);
}

init();
