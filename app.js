const candidates = [
  {
    id: "v001",
    title: "多目标精排中的校准与融合",
    author: "算法工程札记",
    category: "算法",
    ctr: 0.82,
    interest: 0.91,
    freshness: 0.42,
    conversion: 0.38,
    fatigue: 0.08
  },
  {
    id: "v002",
    title: "短视频重排：类目打散的工程实现",
    author: "推荐系统实践",
    category: "算法",
    ctr: 0.78,
    interest: 0.86,
    freshness: 0.58,
    conversion: 0.34,
    fatigue: 0.12
  },
  {
    id: "v003",
    title: "周末城市骑行路线",
    author: "城市生活研究所",
    category: "生活",
    ctr: 0.66,
    interest: 0.52,
    freshness: 0.73,
    conversion: 0.22,
    fatigue: 0.03
  },
  {
    id: "v004",
    title: "新款降噪耳机横评",
    author: "数码观察",
    category: "消费",
    ctr: 0.74,
    interest: 0.63,
    freshness: 0.77,
    conversion: 0.83,
    fatigue: 0.06
  },
  {
    id: "v005",
    title: "实时特征平台的延迟治理",
    author: "算法工程札记",
    category: "工程",
    ctr: 0.69,
    interest: 0.88,
    freshness: 0.49,
    conversion: 0.45,
    fatigue: 0.18
  },
  {
    id: "v006",
    title: "三十分钟高蛋白晚餐",
    author: "厨房效率手册",
    category: "美食",
    ctr: 0.58,
    interest: 0.47,
    freshness: 0.88,
    conversion: 0.29,
    fatigue: 0.02
  },
  {
    id: "v007",
    title: "A/B 实验平台的指标防抖",
    author: "增长实验室",
    category: "增长",
    ctr: 0.71,
    interest: 0.76,
    freshness: 0.62,
    conversion: 0.64,
    fatigue: 0.11
  },
  {
    id: "v008",
    title: "露营装备清单：从入门到轻量化",
    author: "户外日常",
    category: "生活",
    ctr: 0.55,
    interest: 0.44,
    freshness: 0.81,
    conversion: 0.51,
    fatigue: 0.05
  },
  {
    id: "v009",
    title: "召回粗排精排的链路拆解",
    author: "推荐系统实践",
    category: "算法",
    ctr: 0.86,
    interest: 0.94,
    freshness: 0.36,
    conversion: 0.41,
    fatigue: 0.2
  },
  {
    id: "v010",
    title: "适合通勤的轻办公背包",
    author: "数码观察",
    category: "消费",
    ctr: 0.61,
    interest: 0.5,
    freshness: 0.7,
    conversion: 0.78,
    fatigue: 0.04
  }
];

const scenarioPresets = {
  interest: {
    label: "兴趣强化",
    weights: { ctr: 42, interest: 30, freshness: 16, explore: 12 },
    conversionWeight: 0.12
  },
  fresh: {
    label: "新鲜度优先",
    weights: { ctr: 32, interest: 20, freshness: 34, explore: 18 },
    conversionWeight: 0.14
  },
  commerce: {
    label: "转化优先",
    weights: { ctr: 34, interest: 18, freshness: 12, explore: 8 },
    conversionWeight: 0.36
  }
};

const controls = {
  scenario: document.querySelector("#scenario-select"),
  ctr: document.querySelector("#ctr-weight"),
  interest: document.querySelector("#interest-weight"),
  freshness: document.querySelector("#freshness-weight"),
  explore: document.querySelector("#explore-weight")
};

const outputs = {
  ctr: document.querySelector("#ctr-output"),
  interest: document.querySelector("#interest-output"),
  freshness: document.querySelector("#freshness-output"),
  explore: document.querySelector("#explore-output")
};

const rankingBody = document.querySelector("#ranking-body");
const feedList = document.querySelector("#feed-list");
const resetButton = document.querySelector("#reset-button");
const scenarioLabel = document.querySelector("#scenario-label");
const topItemLabel = document.querySelector("#top-item-label");
const diversityScore = document.querySelector("#diversity-score");
const avgScore = document.querySelector("#avg-score");
const exploreCount = document.querySelector("#explore-count");
const constraintText = document.querySelector("#constraint-text");
const deepFMRanker = new window.DeepFMRanker();

function getWeights() {
  const ctr = Number(controls.ctr.value) / 100;
  const interest = Number(controls.interest.value) / 100;
  const freshness = Number(controls.freshness.value) / 100;
  const explore = Number(controls.explore.value) / 100;
  const scenario = scenarioPresets[controls.scenario.value];

  return {
    ctr,
    interest,
    freshness,
    explore,
    conversion: scenario.conversionWeight
  };
}

function applyScenarioPreset() {
  const preset = scenarioPresets[controls.scenario.value];

  controls.ctr.value = preset.weights.ctr;
  controls.interest.value = preset.weights.interest;
  controls.freshness.value = preset.weights.freshness;
  controls.explore.value = preset.weights.explore;
}

function scoreCandidates() {
  const weights = getWeights();

  return candidates
    .map((item) => {
      const modelResult = deepFMRanker.predict({
        ...item,
        scenario: controls.scenario.value
      });
      const businessScore =
        item.ctr * weights.ctr +
        item.interest * weights.interest +
        item.freshness * weights.freshness +
        item.conversion * weights.conversion -
        item.fatigue * 0.18;
      const rawScore = businessScore * 0.55 + modelResult.score * 0.45;

      return {
        ...item,
        deepFMScore: Number(modelResult.score.toFixed(4)),
        deepFMParts: modelResult.parts,
        rankScore: Number(rawScore.toFixed(4))
      };
    })
    .sort((a, b) => b.rankScore - a.rankScore);
}

function rerank(scoredItems) {
  const weights = getWeights();
  const selected = [];
  const remaining = [...scoredItems];
  const categoryCount = new Map();
  const authorCount = new Map();

  while (selected.length < 6 && remaining.length > 0) {
    const position = selected.length + 1;
    const withFinalScore = remaining.map((item) => {
      const reasons = [];
      let finalScore = item.rankScore;
      const categorySeen = categoryCount.get(item.category) || 0;
      const authorSeen = authorCount.get(item.author) || 0;

      if (categorySeen >= 1 && position <= 4) {
        finalScore -= 0.18;
        reasons.push({ type: "penalty", text: "类目打散" });
      }

      if (authorSeen >= 1) {
        finalScore -= 0.12;
        reasons.push({ type: "penalty", text: "作者去重" });
      }

      if (item.fatigue > 0.15) {
        finalScore -= item.fatigue;
        reasons.push({ type: "penalty", text: "疲劳惩罚" });
      }

      if (position === 4 && item.freshness > 0.76) {
        finalScore += weights.explore;
        reasons.push({ type: "boost", text: "探索位" });
      }

      if (selected.length >= 3 && categoryCount.size < 3 && !categoryCount.has(item.category)) {
        finalScore += 0.08;
        reasons.push({ type: "boost", text: "多样性补偿" });
      }

      return {
        ...item,
        finalScore: Number(finalScore.toFixed(4)),
        reasons
      };
    });

    withFinalScore.sort((a, b) => b.finalScore - a.finalScore);
    const chosen = withFinalScore[0];
    selected.push(chosen);
    categoryCount.set(chosen.category, (categoryCount.get(chosen.category) || 0) + 1);
    authorCount.set(chosen.author, (authorCount.get(chosen.author) || 0) + 1);

    const chosenIndex = remaining.findIndex((item) => item.id === chosen.id);
    remaining.splice(chosenIndex, 1);
  }

  return selected;
}

function createBar(value) {
  const wrapper = document.createElement("div");
  wrapper.className = "bar-cell";

  const score = document.createElement("span");
  score.textContent = value.toFixed(2);

  const bar = document.createElement("span");
  bar.className = "bar";

  const fill = document.createElement("span");
  fill.style.width = `${Math.round(value * 100)}%`;
  bar.append(fill);
  wrapper.append(score, bar);

  return wrapper;
}

function renderRanking(scoredItems) {
  rankingBody.replaceChildren();

  scoredItems.forEach((item) => {
    const row = document.createElement("tr");

    const titleCell = document.createElement("td");
    const titleWrap = document.createElement("div");
    titleWrap.className = "item-title";
    const title = document.createElement("strong");
    title.textContent = item.title;
    const author = document.createElement("small");
    author.textContent = item.author;
    titleWrap.append(title, author);
    titleCell.append(titleWrap);

    const categoryCell = document.createElement("td");
    const category = document.createElement("span");
    category.className = "pill";
    category.textContent = item.category;
    categoryCell.append(category);

    const ctrCell = document.createElement("td");
    ctrCell.append(createBar(item.ctr));

    const interestCell = document.createElement("td");
    interestCell.append(createBar(item.interest));

    const deepFMCell = document.createElement("td");
    deepFMCell.className = "score";
    deepFMCell.title = `linear ${item.deepFMParts.linear.toFixed(3)}, fm ${item.deepFMParts.fm.toFixed(3)}, dnn ${item.deepFMParts.dnn.toFixed(3)}`;
    deepFMCell.textContent = item.deepFMScore.toFixed(4);

    const scoreCell = document.createElement("td");
    scoreCell.className = "score";
    scoreCell.textContent = item.rankScore.toFixed(4);

    row.append(titleCell, categoryCell, ctrCell, interestCell, deepFMCell, scoreCell);
    rankingBody.append(row);
  });
}

function renderFeed(rerankedItems) {
  feedList.replaceChildren();

  rerankedItems.forEach((item) => {
    const feedItem = document.createElement("li");
    const isExplore = item.reasons.some((reason) => reason.text === "探索位");
    feedItem.className = `feed-item${isExplore ? " explore" : ""}`;

    const main = document.createElement("div");
    main.className = "feed-main";

    const title = document.createElement("h3");
    title.textContent = item.title;

    const meta = document.createElement("div");
    meta.className = "feed-meta";
    ["最终分 " + item.finalScore.toFixed(4), item.category, item.author].forEach((text) => {
      const tag = document.createElement("span");
      tag.className = "pill";
      tag.textContent = text;
      meta.append(tag);
    });

    const reasons = document.createElement("div");
    reasons.className = "reason-list";
    const reasonList = item.reasons.length > 0 ? item.reasons : [{ type: "boost", text: "保持精排顺序" }];
    reasonList.forEach((reason) => {
      const tag = document.createElement("span");
      tag.className = `reason ${reason.type}`;
      tag.textContent = reason.text;
      reasons.append(tag);
    });

    main.append(title, meta, reasons);
    feedItem.append(main);
    feedList.append(feedItem);
  });
}

function renderDiagnostics(rerankedItems) {
  const categories = new Set(rerankedItems.map((item) => item.category));
  const average = rerankedItems.reduce((sum, item) => sum + item.finalScore, 0) / rerankedItems.length;
  const exploreItems = rerankedItems.filter((item) => item.reasons.some((reason) => reason.text === "探索位"));
  const topItem = rerankedItems[0];

  diversityScore.textContent = `${categories.size}/${rerankedItems.length}`;
  avgScore.textContent = average.toFixed(3);
  exploreCount.textContent = exploreItems.length;
  topItemLabel.textContent = `Top item: ${topItem.title}`;
  constraintText.textContent = "前 4 位避免同类目连续霸榜；同作者降权；疲劳内容降权；第 4 位允许新鲜内容探索。";
}

function renderOutputs() {
  outputs.ctr.textContent = Number(controls.ctr.value / 100).toFixed(2);
  outputs.interest.textContent = Number(controls.interest.value / 100).toFixed(2);
  outputs.freshness.textContent = Number(controls.freshness.value / 100).toFixed(2);
  outputs.explore.textContent = `${controls.explore.value}%`;
  scenarioLabel.textContent = scenarioPresets[controls.scenario.value].label;
}

function render() {
  const scoredItems = scoreCandidates();
  const rerankedItems = rerank(scoredItems);

  renderOutputs();
  renderRanking(scoredItems);
  renderFeed(rerankedItems);
  renderDiagnostics(rerankedItems);
}

Object.values(controls).forEach((control) => {
  control.addEventListener("input", render);
});

controls.scenario.addEventListener("change", () => {
  applyScenarioPreset();
  render();
});

resetButton.addEventListener("click", () => {
  applyScenarioPreset();
  render();
});

applyScenarioPreset();
render();
