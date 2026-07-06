class DeepFMRanker {
  constructor() {
    this.embeddingDim = 4;
    this.linearBias = -0.18;
    this.dnnBias = [0.12, -0.08, 0.05, 0.02, -0.04, 0.07];
    this.dnnOutputWeights = [0.28, -0.16, 0.22, 0.18, -0.12, 0.2];
  }

  predict(sample) {
    const features = this.buildFeatures(sample);
    const linearLogit = this.linear(features);
    const fmLogit = this.fm(features);
    const dnnLogit = this.dnn(features);
    const logit = this.linearBias + linearLogit + fmLogit + dnnLogit;

    return {
      score: this.sigmoid(logit),
      parts: {
        linear: linearLogit,
        fm: fmLogit,
        dnn: dnnLogit
      }
    };
  }

  buildFeatures(sample) {
    return [
      { name: "ctr", value: sample.ctr },
      { name: "interest", value: sample.interest },
      { name: "freshness", value: sample.freshness },
      { name: "conversion", value: sample.conversion },
      { name: "fatigue", value: -sample.fatigue },
      { name: `category:${sample.category}`, value: 1 },
      { name: `author:${sample.author}`, value: 1 },
      { name: `scenario:${sample.scenario}`, value: 1 }
    ];
  }

  linear(features) {
    return features.reduce((sum, feature) => {
      return sum + this.weight(feature.name, "linear") * feature.value;
    }, 0);
  }

  fm(features) {
    let interaction = 0;

    for (let dim = 0; dim < this.embeddingDim; dim += 1) {
      let sum = 0;
      let squareSum = 0;

      features.forEach((feature) => {
        const value = this.embedding(feature.name, dim) * feature.value;
        sum += value;
        squareSum += value * value;
      });

      interaction += 0.5 * (sum * sum - squareSum);
    }

    return interaction;
  }

  dnn(features) {
    const dense = features.slice(0, 5).map((feature) => feature.value);
    const pooledEmbedding = Array.from({ length: this.embeddingDim }, (_, dim) => {
      const sum = features.reduce((total, feature) => {
        return total + this.embedding(feature.name, dim) * feature.value;
      }, 0);

      return sum / features.length;
    });

    const input = [...dense, ...pooledEmbedding];
    const hidden = this.dnnBias.map((bias, neuronIndex) => {
      const activation = input.reduce((sum, value, inputIndex) => {
        return sum + value * this.weight(`dnn:${neuronIndex}:${inputIndex}`, "hidden");
      }, bias);

      return Math.max(0, activation);
    });

    return hidden.reduce((sum, value, index) => {
      return sum + value * this.dnnOutputWeights[index];
    }, 0);
  }

  embedding(name, dim) {
    return this.weight(`${name}:emb:${dim}`, "embedding") * 0.32;
  }

  weight(name, namespace) {
    const hash = this.hash(`${namespace}:${name}`);
    return (hash / 2147483647 - 0.5) * 2;
  }

  hash(text) {
    let value = 2166136261;

    for (let index = 0; index < text.length; index += 1) {
      value ^= text.charCodeAt(index);
      value = Math.imul(value, 16777619);
    }

    return value >>> 1;
  }

  sigmoid(value) {
    return 1 / (1 + Math.exp(-value));
  }
}

window.DeepFMRanker = DeepFMRanker;
