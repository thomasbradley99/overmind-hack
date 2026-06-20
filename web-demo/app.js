function pct(v) { return (v * 100).toFixed(1) + '%'; }
function cls(c) { return c ? 'correct' : 'incorrect'; }

const data = evaluationData;

// 1. Model ranking table
const tbody = document.getElementById('model-ranking-body');
data.modelSummaries.forEach((m, i) => {
  const tr = document.createElement('tr');
  if (i < 3) tr.className = 'best';
  if (m.f1 === 0) tr.className = 'worst';
  tr.innerHTML = `
    <td>${i+1}</td>
    <td>${m.model}</td>
    <td>${m.config}</td>
    <td>${m.size}</td>
    <td><strong>${pct(m.f1)}</strong></td>
    <td>${pct(m.accuracy)}</td>
    <td>${pct(m.precision)}</td>
    <td>${pct(m.recall)}</td>
    <td>${pct(m.team_accuracy)}</td>
    <td>${m.avg_latency.toFixed(1)}s</td>`;
  tbody.appendChild(tr);
});

// 2. Clip selector
const clipSelector = document.getElementById('clip-selector');
data.clips.forEach(c => {
  const btn = document.createElement('button');
  btn.className = `clip-btn ${c.type}-clip`;
  btn.textContent = c.name;
  btn.onclick = () => showClip(c.name, btn);
  clipSelector.appendChild(btn);
});

function showClip(clipName, btn) {
  document.querySelectorAll('.clip-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const clip = data.clips.find(c => c.name === clipName);
  document.getElementById('clip-details').style.display = 'block';
  document.getElementById('selected-truth').textContent = clip.truth.toUpperCase();
  document.getElementById('selected-truth').className = `badge ${clip.truth}`;
  document.getElementById('selected-team').textContent = clip.team ? `Team: ${clip.team}` : '';
  
  const perClip = data.perClipResults[clipName];
  const tbl = document.getElementById('per-clip-table');
  tbl.innerHTML = '';
  Object.entries(perClip).forEach(([model, r]) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${model.split(' (')[0]}</td>
      <td>${model.split('(')[1]?.replace(')', '') || ''}</td>
      <td class="${cls(r.goal_correct)}">${r.pred}</td>
      <td>${r.pred_team || '-'}</td>
      <td class="${cls(r.goal_correct)}">${r.goal_correct ? 'Y' : 'N'}</td>
      <td class="${r.pred === 'goal' && r.team_correct ? 'correct' : r.pred === 'goal' && !r.team_correct ? 'incorrect' : ''}">${r.pred === 'goal' ? (r.team_correct ? 'Y' : 'N') : '-'}</td>`;
    tr.style.cursor = 'pointer';
    tr.onclick = () => showRaw(clipName, model, r);
    tbl.appendChild(tr);
  });
  // Show raw of first model
  const firstModel = Object.keys(perClip)[0];
  showRaw(clipName, firstModel, perClip[firstModel]);
}

function showRaw(clip, model, r) {
  document.getElementById('raw-output').textContent = `Model: ${model}\nClip: ${clip}\nPrediction: ${r.pred}\nTeam: ${r.pred_team || 'N/A'}\nLatency: ${r.latency.toFixed(1)}s\n\nRaw Output:\n${r.raw}`;
}

// 3. Bar charts
function drawGoalChart() {
  const chart = document.getElementById('goal-chart');
  const sorted = [...data.modelSummaries].sort((a, b) => b.f1 - a.f1);
  const maxVal = Math.max(...sorted.map(m => m.f1));
  sorted.forEach(m => {
    const g = document.createElement('div');
    g.className = 'bar-group';
    const f1h = (m.f1 / maxVal) * 180;
    const ph = (m.precision / maxVal) * 180;
    const rh = (m.recall / maxVal) * 180;
    g.innerHTML = `
      <div style="display:flex;align-items:flex-end;gap:2px;width:100%;height:180px">
        <div class="bar f1-bar" style="height:${f1h}px;flex:1"><span class="bar-value">${pct(m.f1)}</span></div>
        <div class="bar prec-bar" style="height:${ph}px;flex:1"><span class="bar-value">${pct(m.precision)}</span></div>
        <div class="bar rec-bar" style="height:${rh}px;flex:1"><span class="bar-value">${pct(m.recall)}</span></div>
      </div>
      <div class="bar-label" title="${m.model} ${m.config}">${m.model}</div>`;
    chart.appendChild(g);
  });
}

function drawTeamChart() {
  const chart = document.getElementById('team-chart');
  const sorted = [...data.modelSummaries].sort((a, b) => b.team_accuracy - a.team_accuracy);
  const maxVal = Math.max(...sorted.map(m => m.team_accuracy));
  sorted.forEach(m => {
    const g = document.createElement('div');
    g.className = 'bar-group';
    const h = m.team_accuracy > 0 ? (m.team_accuracy / maxVal) * 180 : 0;
    g.innerHTML = `
      <div class="bar" style="height:${h}px;background:var(--accent2);width:100%"><span class="bar-value">${pct(m.team_accuracy)}</span></div>
      <div class="bar-label" title="${m.model} ${m.config}">${m.model}</div>`;
    chart.appendChild(g);
  });
}

// 4. Confusion matrices
const cmContainer = document.getElementById('confusion-matrices');
data.modelSummaries.forEach(m => {
  const div = document.createElement('div');
  div.className = 'card';
  div.style.background = 'var(--bg)';
  div.innerHTML = `
    <h3>${m.model}</h3>
    <div style="font-size:0.75rem;color:var(--text2);margin-bottom:0.5rem">${m.config}</div>
    <div class="confusion">
      <div></div><div class="confusion-header">Pred Goal</div><div class="confusion-header">Pred Not-Goal</div>
      <div class="confusion-header">Truth Goal</div><div class="confusion-cell tp">${m.tp}</div><div class="confusion-cell fn">${m.fn}</div>
      <div class="confusion-header">Truth Not-Goal</div><div class="confusion-cell fp">${m.fp}</div><div class="confusion-cell tn">${m.tn}</div>
    </div>
    <div style="margin-top:0.5rem;font-size:0.75rem;text-align:center">
      F1: ${pct(m.f1)} | Acc: ${pct(m.accuracy)} | Lat: ${m.avg_latency.toFixed(1)}s
    </div>`;
  cmContainer.appendChild(div);
});

// 5. Model checkboxes
const modelCheckboxes = document.getElementById('model-checkboxes');
data.modelSummaries.forEach((m, i) => {
  const lbl = document.createElement('label');
  lbl.style.cssText = 'display:flex;align-items:center;gap:0.25rem;cursor:pointer;font-size:0.8rem';
  lbl.innerHTML = `<input type="checkbox" value="${m.model} (${m.config})" ${i < 5 ? 'checked' : ''}> ${m.model} ${m.config}`;
  modelCheckboxes.appendChild(lbl);
});

const runClipCheckboxes = document.getElementById('run-clip-checkboxes');
data.clips.forEach(c => {
  const lbl = document.createElement('label');
  lbl.style.cssText = 'display:flex;align-items:center;gap:0.25rem;cursor:pointer;font-size:0.8rem';
  lbl.innerHTML = `<input type="checkbox" value="${c.name}" checked> ${c.name}`;
  runClipCheckboxes.appendChild(lbl);
});

function runEvaluation() {
  const selectedModels = Array.from(modelCheckboxes.querySelectorAll('input:checked')).map(i => i.value);
  const selectedClips = Array.from(runClipCheckboxes.querySelectorAll('input:checked')).map(i => i.value);
  
  if (!selectedModels.length || !selectedClips.length) {
    alert('Select at least one model and one clip');
    return;
  }
  
  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.innerHTML = 'Running... <span class="spinner"></span>';
  
  // Simulate run
  setTimeout(() => {
    const results = document.getElementById('run-results');
    results.style.display = 'block';
    
    // Aggregate
    const aggBody = document.getElementById('run-aggregate-table').querySelector('tbody');
    aggBody.innerHTML = '';
    selectedModels.forEach(m => {
      const summary = data.modelSummaries.find(s => `${s.model} (${s.config})` === m);
      if (!summary) return;
      // Filter to selected clips
      let tp = 0, fn = 0, fp = 0, tn = 0;
      selectedClips.forEach(c => {
        const r = data.perClipResults[c]?.[m];
        if (!r) return;
        const truth = data.clips.find(cl => cl.name === c).truth;
        if (truth === 'goal' && r.pred === 'goal') tp++;
        else if (truth === 'goal' && r.pred !== 'goal') fn++;
        else if (truth === 'not_goal' && r.pred === 'goal') fp++;
        else if (truth === 'not_goal' && r.pred !== 'goal') tn++;
      });
      const prec = tp + fp ? tp / (tp + fp) : 0;
      const rec = tp + fn ? tp / (tp + fn) : 0;
      const f1 = prec + rec ? 2 * prec * rec / (prec + rec) : 0;
      const acc = selectedClips.length ? (tp + tn) / selectedClips.length : 0;
      const teamTotal = selectedClips.filter(c => {
        const r = data.perClipResults[c]?.[m];
        return data.clips.find(cl => cl.name === c).truth === 'goal' && r?.pred === 'goal';
      }).length;
      const teamCorrect = selectedClips.filter(c => {
        const r = data.perClipResults[c]?.[m];
        return data.clips.find(cl => cl.name === c).truth === 'goal' && r?.pred === 'goal' && r?.team_correct;
      }).length;
      const teamAcc = teamTotal ? teamCorrect / teamTotal : 0;
      
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${m.split('(')[0]}</td><td>${pct(f1)}</td><td>${pct(acc)}</td><td>${pct(prec)}</td><td>${pct(rec)}</td><td>${pct(teamAcc)}</td>`;
      aggBody.appendChild(tr);
    });
    
    // Confusion matrix
    let totalTP = 0, totalFN = 0, totalFP = 0, totalTN = 0;
    selectedClips.forEach(c => {
      selectedModels.forEach(m => {
        const r = data.perClipResults[c]?.[m];
        if (!r) return;
        const truth = data.clips.find(cl => cl.name === c).truth;
        if (truth === 'goal' && r.pred === 'goal') totalTP++;
        else if (truth === 'goal' && r.pred !== 'goal') totalFN++;
        else if (truth === 'not_goal' && r.pred === 'goal') totalFP++;
        else if (truth === 'not_goal' && r.pred !== 'goal') totalTN++;
      });
    });
    document.getElementById('run-confusion').innerHTML = `
      <div class="confusion">
        <div></div><div class="confusion-header">Pred Goal</div><div class="confusion-header">Pred Not-Goal</div>
        <div class="confusion-header">Truth Goal</div><div class="confusion-cell tp">${totalTP}</div><div class="confusion-cell fn">${totalFN}</div>
        <div class="confusion-header">Truth Not-Goal</div><div class="confusion-cell fp">${totalFP}</div><div class="confusion-cell tn">${totalTN}</div>
      </div>`;
    
    // Per-clip table
    const pcHeader = document.getElementById('run-per-clip-header');
    pcHeader.innerHTML = selectedModels.map(m => `<th>${m.split('(')[0].trim()}</th>`).join('');
    const pcBody = document.getElementById('run-per-clip-table').querySelector('tbody');
    pcBody.innerHTML = '';
    selectedClips.forEach(c => {
      const tr = document.createElement('tr');
      const truth = data.clips.find(cl => cl.name === c).truth;
      let cells = selectedModels.map(m => {
        const r = data.perClipResults[c]?.[m];
        if (!r) return '<td>-</td>';
        const ok = r.goal_correct;
        return `<td class="${cls(ok)}">${r.pred}${r.pred === 'goal' && r.pred_team ? '<br>' + r.pred_team : ''}</td>`;
      }).join('');
      tr.innerHTML = `<td>${c}</td><td class="${truth === 'goal' ? 'correct' : 'incorrect'}">${truth}</td>${cells}`;
      pcBody.appendChild(tr);
    });
    
    btn.disabled = false;
    btn.textContent = 'Run Evaluation';
  }, 800);
}

// ENSEMBLE RENDERING
function renderEnsembleTable() {
  const tbody = document.getElementById('ensemble-table-body');
  if (!tbody) return;
  const strategies = ensembleResults.strategies;
  const names = {
    "or": "OR (Union) — Best",
    "and": "AND (Intersection)",
    "cascade": "Cascade (smolvlm2 → moondream)",
    "smolvlm2": "smolvlm2 alone (baseline)",
    "moondream": "moondream alone (baseline)"
  };
  
  Object.entries(strategies).forEach(([key, s]) => {
    const tr = document.createElement('tr');
    if (key === ensembleResults.best_strategy) {
      tr.style.background = 'rgba(56, 189, 248, 0.1)';
      tr.style.borderLeft = '3px solid var(--accent)';
    }
    if (key === 'smolvlm2' || key === 'moondream') {
      tr.style.opacity = '0.7';
    }
    tr.innerHTML = `
      <td><strong>${names[key]}</strong></td>
      <td><strong>${pct(s.f1)}</strong></td>
      <td>${pct(s.precision)}</td>
      <td>${pct(s.recall)}</td>
      <td>${pct(s.accuracy)}</td>
      <td>${pct(s.team_accuracy)}</td>
      <td>${s.tp}</td>
      <td>${s.fp}</td>
      <td>${s.fn}</td>
      <td>${s.tn}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderEnsembleClipTable() {
  const tbody = document.getElementById('ensemble-clip-body');
  if (!tbody) return;
  
  const smolvlm2Key = "smolvlm2-2.2b (56px, 23 frames, 2fps)";
  const moondreamKey = "moondream:1.8b (224px, 1 frame, direct prompt)";
  
  data.clips.forEach(c => {
    const clipData = data.perClipResults[c.name];
    if (!clipData) return;
    const s = clipData[smolvlm2Key];
    const m = clipData[moondreamKey];
    if (!s || !m) return;
    
    // OR ensemble: if either says goal, it's a goal
    const pred = (s.pred === "goal" || m.pred === "goal") ? "goal" : "not_goal";
    const team = (m.pred === "goal" && m.team) ? m.team : (s.pred === "goal" && s.team) ? s.team : null;
    
    const g_ok = c.truth === pred;
    const t_ok = c.truth === "goal" && pred === "goal" && team === c.team;
    
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${c.name}</td>
      <td class="${c.truth === 'goal' ? 'correct' : 'incorrect'}">${c.truth}</td>
      <td class="${cls(s.goal_correct)}">${s.pred}${s.team ? '<br>' + s.team : ''}</td>
      <td class="${cls(m.goal_correct)}">${m.pred}${m.team ? '<br>' + m.team : ''}</td>
      <td class="${cls(g_ok)}"><strong>${pred}</strong>${team ? '<br>' + team : ''}</td>
      <td class="${t_ok ? 'correct' : pred === 'goal' && c.truth === 'goal' ? 'incorrect' : ''}">${team || '-'}</td>
      <td class="${cls(g_ok)}">${g_ok ? '✓' : '✗'}</td>
    `;
    tbody.appendChild(tr);
  });
}

// Initialize
drawGoalChart();
drawTeamChart();
renderEnsembleTable();
renderEnsembleClipTable();
// Select first clip by default
showClip(data.clips[0].name, clipSelector.children[0]);

// --- Live Demo Analysis ---
let demoStrategy = 'or';

function setDemoStrategy(strategy) {
  demoStrategy = strategy;
  document.querySelectorAll('.strategy-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.strategy === strategy);
    btn.style.background = btn.dataset.strategy === strategy ? 'var(--accent)' : '';
    btn.style.color = btn.dataset.strategy === strategy ? 'white' : '';
  });
}

function ensembleDecision(s, m, mode) {
  if (mode === 'or') {
    if (s.pred === 'goal' || m.pred === 'goal') {
      return {
        pred: 'goal',
        team: (m.pred === 'goal' && m.team) ? m.team : s.team,
        confidence: (s.pred === 'goal' && m.pred === 'goal') ? 'high' : 'medium',
        rationale: 'At least one model detected a goal. Union strategy maximizes recall.'
      };
    }
    return { pred: 'not_goal', team: null, confidence: 'high', rationale: 'Both models agree: no goal.' };
  }
  if (mode === 'and') {
    if (s.pred === 'goal' && m.pred === 'goal') {
      return { pred: 'goal', team: m.team || s.team, confidence: 'high', rationale: 'Both models agree on goal.' };
    }
    return { pred: 'not_goal', team: null, confidence: 'high', rationale: 'Models disagree or both say no goal. Conservative.' };
  }
  if (mode === 'cascade') {
    if (s.pred === 'goal' && m.pred === 'goal') {
      return { pred: 'goal', team: m.team || s.team, confidence: 'high', rationale: 'smolvlm2 detected goal, moondream confirmed.' };
    }
    if (s.pred === 'goal' && m.pred !== 'goal') {
      return { pred: 'not_goal', team: null, confidence: 'medium', rationale: 'smolvlm2 detected goal but moondream did not confirm. Vetoed.' };
    }
    return { pred: 'not_goal', team: null, confidence: 'high', rationale: 'smolvlm2 did not detect a goal.' };
  }
  return { pred: 'not_goal', team: null, confidence: 'low', rationale: 'Unknown strategy.' };
}

function runDemoAnalysis() {
  const clipName = document.getElementById('demo-clip-select').value;
  const d = demoData[clipName];
  if (!d) return;

  const btn = document.getElementById('demo-run-btn');
  const resultsDiv = document.getElementById('demo-results');
  const verdictDiv = document.getElementById('demo-verdict');

  btn.disabled = true;
  btn.textContent = 'Analyzing...';
  resultsDiv.style.display = 'block';
  verdictDiv.style.display = 'none';

  // Reset state
  ['smolvlm2', 'moondream', 'ensemble'].forEach(key => {
    document.getElementById('demo-' + key + '-status').textContent = 'Waiting...';
    document.getElementById('demo-' + key + '-result').style.display = 'none';
  });

  // Step 1: smolvlm2 (fast, ~0.7-1.0s)
  setTimeout(() => {
    document.getElementById('demo-smolvlm2-status').textContent = 'Running on macOS Node A (MLX GPU)...';
  }, 100);

  setTimeout(() => {
    const s = d.smolvlm2;
    document.getElementById('demo-smolvlm2-status').textContent = 'Complete';
    document.getElementById('demo-smolvlm2-result').style.display = 'block';
    document.getElementById('demo-smolvlm2-pred').querySelector('.value').textContent = s.pred;
    document.getElementById('demo-smolvlm2-pred').querySelector('.value').style.color = s.pred === 'goal' ? 'var(--goal)' : 'var(--not-goal)';
    document.getElementById('demo-smolvlm2-team').querySelector('.value').textContent = s.team || '-';
    document.getElementById('demo-smolvlm2-latency').querySelector('.value').textContent = s.latency.toFixed(1) + 's';
    document.getElementById('demo-smolvlm2-raw').textContent = s.raw;
  }, 900);

  // Step 2: moondream (~1.0-1.3s)
  setTimeout(() => {
    document.getElementById('demo-moondream-status').textContent = 'Running on Linux Node B (CPU)...';
  }, 100);

  setTimeout(() => {
    const m = d.moondream;
    document.getElementById('demo-moondream-status').textContent = 'Complete';
    document.getElementById('demo-moondream-result').style.display = 'block';
    document.getElementById('demo-moondream-pred').querySelector('.value').textContent = m.pred;
    document.getElementById('demo-moondream-pred').querySelector('.value').style.color = m.pred === 'goal' ? 'var(--goal)' : 'var(--not-goal)';
    document.getElementById('demo-moondream-team').querySelector('.value').textContent = m.team || '-';
    document.getElementById('demo-moondream-latency').querySelector('.value').textContent = m.latency.toFixed(1) + 's';
    document.getElementById('demo-moondream-raw').textContent = m.raw;
  }, 1400);

  // Step 3: ensemble decision (after both complete)
  setTimeout(() => {
    const e = ensembleDecision(d.smolvlm2, d.moondream, demoStrategy);
    document.getElementById('demo-ensemble-status').textContent = 'Combining results...';
    document.getElementById('demo-ensemble-result').style.display = 'block';
    document.getElementById('demo-ensemble-pred').querySelector('.value').textContent = e.pred.toUpperCase();
    document.getElementById('demo-ensemble-pred').querySelector('.value').style.color = e.pred === 'goal' ? 'var(--goal)' : 'var(--not-goal)';
    document.getElementById('demo-ensemble-team').querySelector('.value').textContent = e.team || '-';
    document.getElementById('demo-ensemble-confidence').querySelector('.value').textContent = e.confidence;
    document.getElementById('demo-ensemble-rationale').textContent = e.rationale;

    // Verdict
    const g_ok = d.truth === e.pred;
    const t_ok = d.truth === 'goal' && e.pred === 'goal' && e.team === d.truth_team;

    verdictDiv.style.display = 'block';
    document.getElementById('demo-truth').querySelector('.value').textContent = d.truth.toUpperCase();
    document.getElementById('demo-truth').querySelector('.value').style.color = d.truth === 'goal' ? 'var(--goal)' : 'var(--not-goal)';
    document.getElementById('demo-final').querySelector('.value').textContent = e.pred.toUpperCase();
    document.getElementById('demo-final').querySelector('.value').style.color = e.pred === 'goal' ? 'var(--goal)' : 'var(--not-goal)';
    const correctEl = document.getElementById('demo-correct').querySelector('.value');
    correctEl.textContent = g_ok ? 'YES' : 'NO';
    correctEl.style.color = g_ok ? 'var(--goal)' : 'var(--not-goal)';

    btn.disabled = false;
    btn.textContent = 'Run Analysis';
  }, 1800);
}

// Set default strategy active
setDemoStrategy('or');
