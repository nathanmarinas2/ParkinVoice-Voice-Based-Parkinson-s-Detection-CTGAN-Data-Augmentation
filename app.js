(function () {
  const data = window.APP_DATA;

  if (!data) {
    document.body.innerHTML = '<main style="padding:2rem;font-family:sans-serif"><h1>Falta app-data.js</h1><p>Ejecuta analysis_parkinson.py para generar los resultados que alimentan esta app.</p></main>';
    return;
  }

  const formatNumber = (value, digits = 4) => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return 'Pendiente';
    }
    return Number(value).toFixed(digits);
  };

  const create = (tag, className, text) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  };

  const appendStatCard = (container, label, value) => {
    const card = create('article', 'stat-card');
    card.append(create('span', 'stat-card__label', label));
    card.append(create('strong', 'stat-card__value', value));
    container.append(card);
  };

  const fillImage = (id, src) => {
    const image = document.getElementById(id);
    if (!image) return;
    image.src = src;
  };

  document.getElementById('hero-subtitle').textContent = data.project.subtitle;
  document.getElementById('dataset-size').textContent = `${data.dataset.rows} muestras`;
  document.getElementById('feature-count').textContent = `${data.dataset.features} rasgos`;
  document.getElementById('selected-feature-count').textContent = `${data.selectedFeatureCount} características`;

  const statsContainer = document.getElementById('dataset-stats');
  appendStatCard(statsContainer, 'Total de muestras', `${data.dataset.rows}`);
  appendStatCard(statsContainer, 'Variables predictoras', `${data.dataset.features}`);
  appendStatCard(statsContainer, 'Train / test', `${data.dataset.train_rows} / ${data.dataset.test_rows}`);
  appendStatCard(statsContainer, 'Subset final', `${data.selectedFeatureCount} rasgos`);

  const classDistribution = document.getElementById('class-distribution');
  const classRows = [
    { label: 'Clase 0 · Control', value: Number(data.dataset.class_counts['0'] ?? data.dataset.class_counts[0]), color: '#bc6c25' },
    { label: 'Clase 1 · Parkinson', value: Number(data.dataset.class_counts['1'] ?? data.dataset.class_counts[1]), color: '#0f766e' },
  ];
  const totalClassSamples = classRows.reduce((acc, item) => acc + item.value, 0);
  classRows.forEach((item) => {
    const row = create('div', 'class-row');
    row.append(create('strong', '', `${item.label}: ${item.value}`));
    const share = create('span', 'metric-label', `${formatNumber((item.value / totalClassSamples) * 100, 1)}% del dataset`);
    row.append(share);
    const track = create('div', 'bar-track');
    const fill = create('div', 'bar-fill');
    fill.style.width = `${(item.value / totalClassSamples) * 100}%`;
    fill.style.background = item.color;
    track.append(fill);
    row.append(track);
    classDistribution.append(row);
  });

  const methodologyList = document.getElementById('methodology-list');
  data.methodology.forEach((step) => {
    methodologyList.append(create('li', '', step));
  });

  const familyGrid = document.getElementById('feature-family-grid');
  data.featureFamilies.forEach((family) => {
    const card = create('article', 'family-card');
    card.append(create('span', 'family-card__label', family.family));
    card.append(create('strong', 'strategy-card__title', `${family.count} variables`));
    card.append(create('p', 'family-card__description', family.description));
    const score = create('p', 'strategy-card__note', `Peso relativo: ${formatNumber(Number(family.share) * 100, 1)}% del subconjunto`);
    card.append(score);
    familyGrid.append(card);
  });

  const topFeatures = document.getElementById('top-features');
  data.topFeatures.forEach((feature) => {
    const chip = create('div', 'chip');
    chip.append(create('span', '', feature.feature));
    chip.append(create('small', '', feature.family));
    topFeatures.append(chip);
  });

  fillImage('figure-feature-ranking', data.figures.featureRanking);
  fillImage('figure-feature-count', data.figures.featureCount);
  fillImage('figure-feature-families', data.figures.featureFamilies);
  fillImage('figure-model-comparison', data.figures.modelComparison);
  fillImage('figure-roc-curves', data.figures.rocCurves);
  fillImage('figure-confusion-matrices', data.figures.confusionMatrices);
  fillImage('figure-synthetic-quality', data.figures.syntheticQuality);
  fillImage('figure-robustness', data.figures.robustness);

  const strategyGrid = document.getElementById('strategy-grid');
  const strategyOrder = [
    ['real', 'Datos reales'],
    ['synthetic', 'CTGAN sintético'],
    ['mixed', 'Real + sintético'],
  ];

  strategyOrder.forEach(([key, label]) => {
    const strategy = data.strategies[key];
    if (!strategy) return;
    const card = create('article', `strategy-card ${key === 'real' ? 'strategy-card--highlight' : ''}`);
    card.append(create('span', 'strategy-card__label', label));
    card.append(create('strong', 'strategy-card__title', strategy.selected_model_by_cv || 'Pendiente'));
    card.append(create('p', 'strategy-card__note', `Modelo seleccionado por CV. Mejor hold-out observado: ${strategy.holdout_best_model || 'Pendiente'}.`));

    const metrics = create('div', 'strategy-metrics');
    [
      ['Balanced accuracy', strategy.selected_metrics?.balanced_accuracy],
      ['Sensibilidad', strategy.selected_metrics?.sensitivity],
      ['Especificidad', strategy.selected_metrics?.specificity],
      ['ROC AUC', strategy.selected_metrics?.roc_auc],
    ].forEach(([metricLabel, metricValue]) => {
      const box = create('div', 'metric-box');
      box.append(create('span', 'metric-label', metricLabel));
      box.append(create('strong', 'metric-value', formatNumber(metricValue)));
      metrics.append(box);
    });
    card.append(metrics);
    strategyGrid.append(card);
  });

  const robustnessContainer = document.getElementById('robustness-summary');
  const robustnessNote = create(
    'p',
    'robustness-note',
    `Semillas evaluadas: ${Array.isArray(data.robustness.seeds) ? data.robustness.seeds.join(', ') : 'Pendiente'}. CTGAN en robustez: ${data.robustness.ctgan_epochs ?? 'Pendiente'} epochs.`
  );
  robustnessContainer.append(robustnessNote);

  if (Array.isArray(data.robustness.summary) && data.robustness.summary.length > 0) {
    const table = create('table', 'robustness-table');
    const thead = create('thead');
    const headerRow = create('tr');
    ['Estrategia', 'Modelo', 'Bal. acc.', 'Sensibilidad', 'Especificidad', 'ROC AUC'].forEach((header) => {
      headerRow.append(create('th', '', header));
    });
    thead.append(headerRow);
    table.append(thead);

    const tbody = create('tbody');
    data.robustness.summary.forEach((row) => {
      const tr = create('tr');
      [
        strategyOrder.find(([key]) => key === row.strategy)?.[1] || row.strategy,
        row.model,
        `${formatNumber(row.balanced_accuracy_mean)} ± ${formatNumber(row.balanced_accuracy_std)}`,
        `${formatNumber(row.sensitivity_mean)} ± ${formatNumber(row.sensitivity_std)}`,
        `${formatNumber(row.specificity_mean)} ± ${formatNumber(row.specificity_std)}`,
        `${formatNumber(row.roc_auc_mean)} ± ${formatNumber(row.roc_auc_std)}`,
      ].forEach((value) => tr.append(create('td', '', value)));
      tbody.append(tr);
    });
    table.append(tbody);
    robustnessContainer.append(table);
  }

  const driftList = document.getElementById('drift-list');
  data.syntheticQuality.topDrift.forEach((item) => {
    const container = create('div', 'drift-item');
    container.append(create('strong', '', `${item.feature} · ${item.family}`));
    container.append(
      create(
        'span',
        'strategy-card__note',
        `KS = ${formatNumber(item.ks_stat)} · Wasserstein normalizado = ${formatNumber(item.wasserstein_normalized)}`
      )
    );
    driftList.append(container);
  });

  const conclusionList = document.getElementById('conclusion-list');
  data.conclusions.forEach((line) => conclusionList.append(create('li', '', line)));
})();
