const svgNamespace = "http://www.w3.org/2000/svg";
const colors = {
  forecast: "#4779a8",
  observed: "#b96243",
  grid: "#ededed",
};
const familyDescriptions = {
  linear: [
    "Linear ridge predicts one point.",
    "A linear map uses the final context-token activation to estimate the expected mean-pooled answer vector. It does not forecast the spread.",
  ],
  mlp: [
    "MLP predicts one point.",
    "A neural network uses the pooled context trace to estimate the expected mean-pooled answer vector. Different actual answers can land around that expectation.",
  ],
  gaussian: [
    "Gaussian forecasts a distribution.",
    "These 64 saved samples come from a Gaussian centered on the frozen MLP forecast, with a fitted covariance and prompt-dependent scale.",
  ],
  flow: [
    "Flow forecasts a distribution.",
    "These 64 saved samples come from a context-conditioned flow. Each blue point is a possible answer activation vector, not a generated answer.",
  ],
};
let demo;
let trajectories;
let selectedExample;
let selectedTrajectories;
let selectedFamily = "flow";
let selectedRollout = 0;
let generatedTokens = 0;
let maximumTokens = 0;
let completedRollouts = 0;
let replayTimer;
let copyTimer;

function svgElement(name, attributes = {}, text) {
  const element = document.createElementNS(svgNamespace, name);
  for (const [attribute, value] of Object.entries(attributes))
    element.setAttribute(attribute, value);
  if (text !== undefined) element.textContent = text;
  return element;
}

function textElement(name, className, text) {
  const element = document.createElement(name);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function getForecast(family = selectedFamily) {
  return selectedExample.forecasts[family === "linear" ? "ridge" : family];
}

function rolloutState(index) {
  const observation = selectedExample.observations[index];
  const trajectory = selectedTrajectories[index];
  const count = Math.min(generatedTokens, observation.tokenCount);
  const complete = count === observation.tokenCount;
  return {
    count,
    complete,
    point: complete ? observation.point : trajectory.points[count - 1],
    text: observation.text.slice(0, count ? trajectory.textEnds[count - 1] : 0),
  };
}

function rolloutTrail(index) {
  const state = rolloutState(index);
  const points = selectedTrajectories[index].points.slice(0, state.count);
  if (state.complete) points[points.length - 1] = state.point;
  return points;
}

function plotBounds() {
  const bounds = {
    x: [...selectedExample.bounds.x],
    y: [...selectedExample.bounds.y],
  };
  if (!generatedTokens) return bounds;
  for (const index of selectedTrajectories.keys()) {
    for (const [x, y] of rolloutTrail(index)) {
      bounds.x[0] = Math.min(bounds.x[0], x);
      bounds.x[1] = Math.max(bounds.x[1], x);
      bounds.y[0] = Math.min(bounds.y[0], y);
      bounds.y[1] = Math.max(bounds.y[1], y);
    }
  }
  return bounds;
}

function setPressedButtons(container, attribute, value) {
  for (const button of container.querySelectorAll("button")) {
    button.setAttribute(
      "aria-pressed",
      String(button.dataset[attribute] === value),
    );
  }
}

function plotCoordinates(plot) {
  const bounds = plotBounds();
  const center = [
    (bounds.x[0] + bounds.x[1]) / 2,
    (bounds.y[0] + bounds.y[1]) / 2,
  ];
  const frameWidth = plot.clientWidth || 640;
  const compact = frameWidth < 400;
  const left = 68;
  const right = frameWidth - (compact ? 22 : 42);
  const bottom = compact ? 294 : 344;
  const width = right - left;
  const height = bottom - 34;
  const unitsPerPixel =
    Math.max(
      (bounds.x[1] - bounds.x[0]) / width,
      (bounds.y[1] - bounds.y[0]) / height,
    ) * 1.18;
  const minimum = [
    center[0] - (width * unitsPerPixel) / 2,
    center[1] - (height * unitsPerPixel) / 2,
  ];
  return {
    x: (value) => left + (value - minimum[0]) / unitsPerPixel,
    y: (value) => bottom - (value - minimum[1]) / unitsPerPixel,
    frameWidth,
    left,
    right,
    bottom,
    minimum,
    span: [width * unitsPerPixel, height * unitsPerPixel],
  };
}

function renderActivationPlot() {
  const plot = document.getElementById("activation-plot");
  const coordinates = plotCoordinates(plot);
  plot.setAttribute(
    "viewBox",
    `0 0 ${coordinates.frameWidth} ${coordinates.bottom + 66}`,
  );
  const forecast = getForecast();
  const points = forecast.points || [forecast.point];
  const title = svgElement(
    "title",
    { id: "activation-title" },
    `${selectedFamily === "linear" ? "Linear ridge" : selectedFamily.toUpperCase()} forecast for ${selectedExample.title}`,
  );
  const description = svgElement(
    "desc",
    { id: "activation-description" },
    `${points.length} forecast ${points.length === 1 ? "point" : "samples"}. Token position ${generatedTokens}; ${completedRollouts} of ${selectedExample.observations.length} rollouts complete. All points use the same training-fitted principal components. Axes fit the forecast and rollout paths so far.`,
  );
  plot.replaceChildren(title, description);
  for (let index = 0; index <= 4; index += 1) {
    const xValue = coordinates.minimum[0] + (coordinates.span[0] * index) / 4;
    const yValue = coordinates.minimum[1] + (coordinates.span[1] * index) / 4;
    const x = coordinates.x(xValue);
    const y = coordinates.y(yValue);
    plot.append(
      svgElement("line", {
        x1: x,
        y1: 34,
        x2: x,
        y2: coordinates.bottom,
        stroke: colors.grid,
      }),
    );
    plot.append(
      svgElement("line", {
        x1: coordinates.left,
        y1: y,
        x2: coordinates.right,
        y2: y,
        stroke: colors.grid,
      }),
    );
    plot.append(
      svgElement(
        "text",
        {
          x,
          y: coordinates.bottom + 23,
          "text-anchor": "middle",
          class: "chart-axis",
        },
        xValue.toFixed(1),
      ),
    );
    plot.append(
      svgElement(
        "text",
        {
          x: coordinates.left - 13,
          y: y + 4,
          "text-anchor": "end",
          class: "chart-axis",
        },
        yValue.toFixed(1),
      ),
    );
  }
  plot.append(
    svgElement(
      "text",
      {
        x: (coordinates.left + coordinates.right) / 2,
        y: coordinates.bottom + 53,
        "text-anchor": "middle",
        class: "chart-axis",
      },
      "Principal component 1",
    ),
  );
  plot.append(
    svgElement(
      "text",
      {
        transform: `translate(14 ${(34 + coordinates.bottom) / 2}) rotate(-90)`,
        "text-anchor": "middle",
        class: "chart-axis",
      },
      "Principal component 2",
    ),
  );

  const trails = svgElement("g", { "aria-hidden": "true" });
  plot.append(trails);
  for (const [index, point] of points.entries()) {
    const circle = svgElement("circle", {
      cx: coordinates.x(point[0]),
      cy: coordinates.y(point[1]),
      r: points.length === 1 ? 7 : 3.7,
      fill: colors.forecast,
      opacity: points.length === 1 ? 1 : 0.48,
      class: "plot-point",
    });
    circle.append(
      svgElement(
        "title",
        {},
        `${points.length === 1 ? "Point forecast" : `Forecast sample ${index + 1}`}: (${point.map((value) => value.toFixed(3)).join(", ")})`,
      ),
    );
    plot.append(circle);
  }
  if (forecast.mean) {
    const [x, y] = [
      coordinates.x(forecast.mean[0]),
      coordinates.y(forecast.mean[1]),
    ];
    const mean = svgElement("path", {
      d: `M${x - 6},${y} H${x + 6} M${x},${y - 6} V${y + 6}`,
      stroke: colors.forecast,
      "stroke-width": 2.5,
    });
    mean.append(
      svgElement(
        "title",
        {},
        selectedFamily === "gaussian"
          ? "Gaussian analytic mean (the frozen MLP forecast)"
          : "Mean of the 64 displayed flow samples",
      ),
    );
    plot.append(mean);
  }

  const observationsByPoint = new Map();
  for (
    let index = 0;
    generatedTokens > 0 && index < selectedExample.observations.length;
    index += 1
  ) {
    const state = rolloutState(index);
    if (state.count > 1) {
      const trail = rolloutTrail(index).map(
        ([x, y]) => `${coordinates.x(x)},${coordinates.y(y)}`,
      );
      trails.append(
        svgElement("polyline", {
          points: trail.join(" "),
          class: "rollout-trail",
        }),
        svgElement("polyline", {
          points: trail.slice(-32).join(" "),
          class: "rollout-trail rollout-trail-recent",
        }),
      );
    }
    const key = state.point.join(",");
    const group = observationsByPoint.get(key) || {
      point: state.point,
      complete: state.complete,
      indices: [],
    };
    group.indices.push(index + 1);
    observationsByPoint.set(key, group);
  }
  for (const { point, indices, complete } of observationsByPoint.values()) {
    const x = coordinates.x(point[0]);
    const y = coordinates.y(point[1]);
    const circle = svgElement("circle", {
      cx: x,
      cy: y,
      r: complete ? 6 : 4.5,
      opacity: complete ? 1 : 0.75,
      class: "observation-point",
      tabindex: 0,
      role: "button",
      "data-answer-index": indices[0] - 1,
      "aria-label": `Read rollout ${indices[0]}${complete ? ", complete" : ` at token ${generatedTokens}`}`,
    });
    circle.append(
      svgElement(
        "title",
        {},
        `Rollout${indices.length > 1 ? "s" : ""} ${indices.join(", ")}: (${point.map((value) => value.toFixed(3)).join(", ")})`,
      ),
    );
    plot.append(circle);
  }
}

function showSamplingStatus(text) {
  const status = document.getElementById("replay-status");
  if (status.textContent !== text) status.textContent = text;
}

function pauseForReading() {
  if (replayTimer) {
    stopReplay();
    showSamplingStatus(
      "Sampling paused while you read. Continue when you’re ready.",
    );
  }
}

function openAnswer(index) {
  selectedRollout = index;
  renderAnswers();
  const reader = document.getElementById("answer-reader");
  reader.scrollTop = replayTimer ? reader.scrollHeight : 0;
  document
    .getElementById("rollout-options")
    .children[index].focus({ preventScroll: true });
}

function renderAnswers() {
  const options = document.getElementById("rollout-options");
  for (const [index, observation] of selectedExample.observations.entries()) {
    let button = options.children[index];
    if (!button) {
      button = textElement("button", "rollout-option", String(index + 1));
      button.type = "button";
      button.dataset.answerIndex = String(index);
      options.append(button);
    }
    const state = rolloutState(index);
    button.setAttribute("aria-pressed", String(index === selectedRollout));
    button.style.setProperty(
      "--rollout-progress",
      `${(100 * state.count) / observation.tokenCount}%`,
    );
    const label = `Rollout ${index + 1} · seed ${observation.seed} · ${state.count} / ${observation.tokenCount} tokens${state.complete ? " · complete" : ""}`;
    button.setAttribute("aria-label", label);
    button.title = label;
  }
  const state = rolloutState(selectedRollout);
  const observation = selectedExample.observations[selectedRollout];
  document.getElementById("answer-label").textContent =
    `Rollout ${String(selectedRollout + 1).padStart(2, "0")}`;
  document.getElementById("answer-progress").textContent =
    `${state.count.toLocaleString()} / ${observation.tokenCount.toLocaleString()} tokens`;
  const placeholder = document.getElementById("answer-placeholder");
  placeholder.hidden = Boolean(state.text);
  placeholder.textContent = state.count
    ? "Generating…"
    : "Sample rollouts to see an answer.";
  const answer = document.getElementById("answer-text");
  if (answer.textContent !== state.text) {
    answer.textContent = state.text;
    if (replayTimer) {
      const reader = document.getElementById("answer-reader");
      reader.scrollTop = reader.scrollHeight;
    }
  }
  document.getElementById("reveal-count").textContent =
    `${completedRollouts} / ${selectedExample.observations.length} complete`;
  document.getElementById("reveal-all").disabled =
    generatedTokens === maximumTokens;
  document.getElementById("reset-replay").disabled = generatedTokens === 0;
  document.getElementById("token-progress").value = generatedTokens;
  document.getElementById("token-count").textContent =
    `${generatedTokens} / ${maximumTokens}`;
}

function stopReplay() {
  clearInterval(replayTimer);
  replayTimer = undefined;
  document.getElementById("play-replay").textContent =
    completedRollouts === selectedExample.observations.length
      ? "Sample rollouts again"
      : generatedTokens > 0
        ? "Continue sampling"
        : "Sample rollouts";
}

function updateReplay() {
  completedRollouts = selectedExample.observations.filter(
    (observation) => generatedTokens >= observation.tokenCount,
  ).length;
  renderActivationPlot();
  renderAnswers();
  const complete = completedRollouts === selectedExample.observations.length;
  if (complete) stopReplay();
  showSamplingStatus(
    complete
      ? "All 16 rollouts complete. Select a point to read its answer."
      : generatedTokens > 0
        ? replayTimer
          ? "Sampling 16 rollouts. The forecast stays fixed."
          : "Paused. Continue sampling or explore with the token slider."
        : "The forecast is ready. Sample rollouts to compare.",
  );
}

function playReplay() {
  if (replayTimer) {
    stopReplay();
    showSamplingStatus("Sampling paused. Continue when you’re ready.");
    return;
  }
  if (generatedTokens === maximumTokens) generatedTokens = 0;
  replayTimer = setInterval(
    () => {
      generatedTokens += 1;
      updateReplay();
    },
    Math.max(30, Math.min(180, 8000 / maximumTokens)),
  );
  document.getElementById("play-replay").textContent = "Pause sampling";
  generatedTokens += 1;
  updateReplay();
}

function renderFamily() {
  setPressedButtons(
    document.getElementById("model-options"),
    "family",
    selectedFamily,
  );
  document.getElementById("forecast-kind").textContent = [
    "linear",
    "mlp",
  ].includes(selectedFamily)
    ? "Point forecast"
    : "Distribution forecast";
  const description = document.getElementById("family-explanation");
  description.replaceChildren(
    textElement("strong", "", familyDescriptions[selectedFamily][0]),
    document.createTextNode(` ${familyDescriptions[selectedFamily][1]}`),
  );
  const forecast = getForecast();
  document.getElementById("projection-note").textContent =
    `Training-fitted PCA · axes fit the forecast and rollout paths so far.${forecast.mean ? (selectedFamily === "gaussian" ? " Blue cross: analytic mean." : " Blue cross: sample mean.") : ""}`;
  renderActivationPlot();
  renderAnswers();
}

function selectExample(id) {
  stopReplay();
  selectedExample =
    demo.examples.find((example) => example.id === id) || demo.examples[0];
  selectedTrajectories = trajectories.examples.find(
    (example) => example.id === selectedExample.id,
  ).rollouts;
  generatedTokens = 0;
  selectedRollout = 0;
  completedRollouts = 0;
  maximumTokens = Math.max(
    ...selectedExample.observations.map(
      (observation) => observation.tokenCount,
    ),
  );
  document.getElementById("token-progress").max = maximumTokens;
  document.getElementById("rollout-options").replaceChildren();
  document.getElementById("answer-reader").scrollTop = 0;
  document.getElementById("example-select").value = selectedExample.id;
  document.getElementById("prompt-text").textContent = selectedExample.prompt;
  document.getElementById("prompt-meta").textContent =
    `${demo.metadata.model} · held-out LMSYS prompt · ${selectedExample.observations.length} independent rollouts · temperature ${demo.metadata.generation.temperature.toFixed(1)}`;
  stopReplay();
  updateReplay();
}

function updateExampleUrl() {
  if (window.location.protocol === "file:") return;
  const url = new URL(window.location.href);
  url.searchParams.set("prompt", selectedExample.id);
  url.searchParams.set("model", selectedFamily);
  window.history.replaceState(null, "", url);
}

function initializeDemo(data, tokenTrajectories) {
  demo = data;
  trajectories = tokenTrajectories;
  const parameters = new URLSearchParams(window.location.search);
  selectedFamily = Object.hasOwn(familyDescriptions, parameters.get("model"))
    ? parameters.get("model")
    : "flow";
  selectedExample =
    demo.examples.find((example) => example.id === parameters.get("prompt")) ||
    demo.examples.find((example) => example.id === "arithmetic") ||
    demo.examples[0];
  const select = document.getElementById("example-select");
  for (const example of demo.examples) {
    const option = textElement(
      "option",
      "",
      `${example.category} · ${example.title}`,
    );
    option.value = example.id;
    select.append(option);
  }
  document.getElementById("demo-loading").hidden = true;
  document.getElementById("demo-content").hidden = false;
  document.getElementById("explorer").setAttribute("aria-busy", "false");
  document.getElementById("demo-provenance").textContent =
    `${demo.metadata.model}, decoder block ${demo.metadata.decoderBlock} (zero-indexed), ${demo.metadata.hiddenDimensions.toLocaleString()} dimensions. Models trained on ${demo.metadata.trainingPrompts.toLocaleString()} prompts. PCA was fitted on ${demo.metadata.projection.fit_rollouts.toLocaleString()} training rollouts, then frozen. Each example has ${demo.metadata.observationsPerPrompt} recorded rollouts from the paper’s supplementary rollout bank.`;
  selectExample(selectedExample.id);
  renderFamily();

  select.addEventListener("change", () => {
    selectExample(select.value);
    updateExampleUrl();
  });
  document
    .getElementById("model-options")
    .addEventListener("click", (event) => {
      const button = event.target.closest("button[data-family]");
      if (!button) return;
      selectedFamily = button.dataset.family;
      renderFamily();
      updateExampleUrl();
    });
  document.getElementById("play-replay").addEventListener("click", playReplay);
  const plot = document.getElementById("activation-plot");
  plot.addEventListener("click", (event) => {
    const point = event.target.closest(".observation-point");
    if (point) openAnswer(Number(point.dataset.answerIndex));
  });
  plot.addEventListener("focusin", (event) => {
    if (
      replayTimer &&
      event.target.matches(".observation-point:focus-visible")
    ) {
      stopReplay();
      showSamplingStatus(
        "Sampling paused. Select a point to read its rollout.",
      );
    }
  });
  plot.addEventListener("keydown", (event) => {
    const point = event.target.closest(".observation-point");
    if (point && ["Enter", " "].includes(event.key)) {
      event.preventDefault();
      openAnswer(Number(point.dataset.answerIndex));
    }
  });
  document.getElementById("reveal-all").addEventListener("click", () => {
    stopReplay();
    generatedTokens = maximumTokens;
    updateReplay();
  });
  document
    .getElementById("token-progress")
    .addEventListener("input", (event) => {
      stopReplay();
      generatedTokens = Number(event.target.value);
      updateReplay();
      stopReplay();
    });
  document
    .getElementById("rollout-options")
    .addEventListener("click", (event) => {
      const button = event.target.closest("button[data-answer-index]");
      if (button) openAnswer(Number(button.dataset.answerIndex));
    });
  const reader = document.getElementById("answer-reader");
  reader.addEventListener("pointerdown", pauseForReading);
  reader.addEventListener("focusin", pauseForReading);
  reader.addEventListener("wheel", pauseForReading, { passive: true });
  document
    .getElementById("reset-replay")
    .addEventListener("click", () => selectExample(selectedExample.id));
  document.getElementById("share-example").addEventListener("click", () => {
    const url = new URL(
      window.location.protocol === "file:"
        ? document.querySelector('link[rel="canonical"]').href
        : window.location.href,
    );
    url.searchParams.set("prompt", selectedExample.id);
    url.searchParams.set("model", selectedFamily);
    url.hash = "explore";
    copyText(url.toString(), "Example link copied.");
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && replayTimer) {
      stopReplay();
      document.getElementById("replay-status").textContent =
        "Sampling paused while the page was hidden.";
    }
  });
}

async function copyText(text, successMessage) {
  const status = document.getElementById("copy-status");
  clearTimeout(copyTimer);
  try {
    await navigator.clipboard.writeText(text);
    status.textContent = successMessage;
  } catch {
    status.textContent =
      "Clipboard unavailable. You can select the citation or copy the URL from your address bar.";
  }
  copyTimer = setTimeout(() => {
    status.textContent = "";
  }, 4500);
}

document
  .getElementById("copy-citation")
  .addEventListener("click", () =>
    copyText(
      document.getElementById("citation").textContent,
      "Citation copied.",
    ),
  );
if (typeof contextAnswerData !== "undefined") {
  initializeDemo(contextAnswerData.demo, contextAnswerData.trajectories);
} else {
  document.getElementById("explorer").setAttribute("aria-busy", "false");
  const message = document.getElementById("demo-loading");
  message.replaceChildren(
    document.createTextNode("The recorded forecasts could not load. "),
  );
  const retry = textElement("button", "text-button", "Try again");
  retry.addEventListener("click", () => window.location.reload());
  message.append(retry);
}

window.addEventListener("resize", () => {
  if (selectedExample) renderActivationPlot();
});
