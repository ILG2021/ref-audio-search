const DEFAULT_WEIGHTS = [1.4, 1.2, 0.8, 0.5, 0.7, 0.35, 0.15];

export function distance(a, b, weights = DEFAULT_WEIGHTS) {
  const scales = [0.35, 0.8, 0.8, 0.08, 0.12, 0.12, 20];
  let sum = 0;
  let weightSum = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const weight = weights[i] ?? 1;
    const delta = (a[i] - b[i]) / scales[i];
    sum += weight * delta * delta;
    weightSum += weight;
  }
  return Math.sqrt(sum / Math.max(weightSum, 1e-9));
}

export function searchByVector(items, query, options = {}) {
  const limit = Number(options.limit || 20);
  const minDuration = Number(options.minDuration || 0);
  const maxDuration = Number(options.maxDuration || Infinity);
  const favorites = options.favorites || new Set();
  return items
    .filter(item => item.duration >= minDuration && item.duration <= maxDuration)
    .map(item => ({ item, distance: distance(query, item.features.prosody.vector) }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, limit)
    .map(({ item, distance: value }) => ({
      id: item.id,
      name: item.name,
      duration: item.duration,
      score: Math.exp(-value),
      features: item.features.prosody,
      favorite: favorites.has(item.id)
    }));
}

function cosine(a, b) {
  if (!a?.length || a.length !== b?.length) return null;
  let dot = 0, aa = 0, bb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; }
  return dot / Math.max(Math.sqrt(aa * bb), 1e-12);
}

const ATTRIBUTE_DIMENSIONS = {
  "慢速": [0, 1, 2, 6], "快速": [0, 1, 2, 6],
  "停顿多": [1, 2], "少停顿": [1, 2],
  "轻声": [3, 4], "有力": [3, 4],
  "克制": [4, 5], "夸张": [4, 5]
};

function prosodySimilarity(prototype, candidate) {
  if (!prototype?.attributes?.length || !candidate?.vector) return null;
  const indices = new Set(prototype.attributes.flatMap(attribute => ATTRIBUTE_DIMENSIONS[attribute] || []));
  if (!indices.size) return null;
  const weights = prototype.vector.map((_, index) => indices.has(index) ? 1 : 0);
  return Math.exp(-distance(prototype.vector, candidate.vector, weights));
}

export function searchByStyleText(items, prototype, options = {}) {
  if (!prototype.attributes.length) return [];
  const favorites = options.favorites || new Set();
  const minDuration = Number(options.minDuration || 0);
  const maxDuration = Number(options.maxDuration || Infinity);
  return items.filter(item => item.duration >= minDuration && item.duration <= maxDuration)
    .map(item => ({ item, score: prosodySimilarity(prototype, item.features.prosody) }))
    .filter(result => result.score != null)
    .sort((a, b) => b.score - a.score)
    .slice(0, Number(options.limit || 20))
    .map(({ item, score }) => ({ id: item.id, name: item.name, duration: item.duration, transcript: item.transcript, score, favorite: favorites.has(item.id), features: item.features.prosody }));
}

export function normalizeContent(text) {
  return String(text || "").normalize("NFKC").toLowerCase().replace(/[\p{P}\p{S}\s]+/gu, "");
}

export function searchByContent(items, query, options = {}) {
  const needle = normalizeContent(query);
  if (!needle) return [];
  const favorites = options.favorites || new Set();
  const minDuration = Number(options.minDuration || 0);
  const maxDuration = Number(options.maxDuration || Infinity);
  return items.filter(item => item.duration >= minDuration && item.duration <= maxDuration)
    .map(item => ({ item, content: normalizeContent(item.transcript) }))
    .filter(({ content }) => content.includes(needle))
    .sort((a, b) => (a.content === needle ? 0 : 1) - (b.content === needle ? 0 : 1) || a.content.length - b.content.length)
    .slice(0, Number(options.limit || 20))
    .map(({ item, content }) => ({ id: item.id, name: item.name, duration: item.duration, transcript: item.transcript, score: content === needle ? 1 : 0.9, favorite: favorites.has(item.id), features: item.features.prosody }));
}

export function searchByModel(items, query, options = {}) {
  const limit = Number(options.limit || 20);
  const minDuration = Number(options.minDuration || 0);
  const maxDuration = Number(options.maxDuration || Infinity);
  const mode = options.mode || "mixed";
  const favorites = options.favorites || new Set();
  return items.filter(item => item.duration >= minDuration && item.duration <= maxDuration).map(item => {
    const style = cosine(query.style, item.features.style?.embedding);
    const emotion = cosine(query.emotion, item.features.emotion?.embedding);
    if (style == null && emotion == null) return null;
    const prosodyScore = mode === "emotion" ? prosodySimilarity(options.prosody, item.features.prosody) : null;
    const emotionScore = prosodyScore == null || emotion == null ? emotion : 0.7 * emotion + 0.3 * prosodyScore;
    const score = mode === "style" ? style : mode === "emotion" ? emotionScore : ((style ?? 0) * 0.65 + (emotion ?? 0) * 0.35);
    return { id: item.id, name: item.name, duration: item.duration, transcript: item.transcript, score, styleScore: style, emotionScore: emotion, prosodyScore, favorite: favorites.has(item.id), features: item.features.prosody };
  }).filter(Boolean).sort((a, b) => b.score - a.score).slice(0, limit);
}

const emotionProfiles = {
  calm:    [0.82, 0.25, 0.45, 0.030, 0.025, 0.035, 6],
  happy:   [0.90, 0.12, 0.25, 0.055, 0.075, 0.070, 5],
  sad:     [0.70, 0.42, 0.65, 0.025, 0.020, 0.030, 7],
  angry:   [0.95, 0.08, 0.22, 0.080, 0.100, 0.085, 4],
  afraid:  [0.88, 0.20, 0.30, 0.050, 0.090, 0.095, 5],
  neutral: [0.84, 0.22, 0.42, 0.040, 0.035, 0.050, 6]
};

const lexicon = [
  ["angry", /愤怒|生气|暴躁|恼火|怒|angry/i],
  ["sad", /悲伤|难过|失望|低落|哭|sad/i],
  ["happy", /开心|高兴|快乐|兴奋|喜悦|happy/i],
  ["afraid", /害怕|恐惧|紧张|不安|惊慌|fear/i],
  ["calm", /平静|克制|温柔|放松|轻声|calm/i]
];

export function textToPrototype(text) {
  const match = lexicon.find(([, pattern]) => pattern.test(text));
  const emotion = match?.[0] || "neutral";
  const vector = [...emotionProfiles[emotion]];
  const attributes = [];
  const withoutNegatedShouting = text.replace(/(?:不要|别|避免|切勿|禁止|不想|不能).{0,4}(?:喊叫|大声|喊|吼叫|吼)/g, "");
  const apply = (pattern, name, fn) => {
    if (pattern.test(text)) { attributes.push(name); fn(vector); }
  };
  apply(/慢速|语速慢|慢语速|缓慢|慢一点|徐徐/i, "慢速", v => { v[0] -= .12; v[1] += .12; v[2] += .18; v[6] += 1.5; });
  apply(/快速|语速快|快语速|急促|快一点|紧凑/i, "快速", v => { v[0] += .08; v[1] -= .08; v[2] -= .12; v[6] -= 1; });
  apply(/停顿多|多停顿|犹豫|断断续续/i, "停顿多", v => { v[1] += .35; v[2] += .25; });
  apply(/少停顿|连贯|一气呵成|流畅/i, "少停顿", v => { v[1] -= .14; v[2] -= .15; });
  apply(/轻声|轻柔|声音小|低声|耳语/i, "轻声", v => { v[3] *= .58; v[4] *= .7; });
  if (/大声|有力|喊|高能量|强烈/i.test(withoutNegatedShouting)) {
    attributes.push("有力"); vector[3] *= 1.55; vector[4] *= 1.4;
  }
  apply(/克制|平缓|平稳|冷静/i, "克制", v => { v[4] *= .62; v[5] *= .75; });
  apply(/夸张|起伏大|戏剧化|富有表现力/i, "夸张", v => { v[4] *= 1.55; v[5] *= 1.3; });
  return { emotion, attributes, vector: vector.map((value, index) => index === 6 ? Math.max(.5, value) : Math.max(0, value)) };
}
