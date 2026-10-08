async function doFetch(path, opts={}){
  const url = path;
  try{
    const res = await fetch(url, {headers: {'Content-Type':'application/json'}, ...opts});
    const txt = await res.text();
    try{ return JSON.parse(txt); }catch(e){ return txt; }
  }catch(e){ return {error: String(e)} }
}

document.getElementById('btn-health').onclick = async ()=>{
  const out = await doFetch('/health');
  document.getElementById('out-health').textContent = JSON.stringify(out, null, 2);
}

document.getElementById('btn-predict').onclick = async ()=>{
  const useLast = document.getElementById('use-last').checked;
  const body = { use_last_row: useLast };
  const out = await doFetch('/predict', { method:'POST', body: JSON.stringify(body) });
  document.getElementById('out-predict').textContent = JSON.stringify(out, null, 2);
}

// Toggle the feature form area
document.getElementById('btn-toggle-form').onclick = ()=>{
  const el = document.getElementById('form-area');
  el.style.display = el.style.display === 'none' || el.style.display === '' ? 'block' : 'none';
}

// Load features list from server and render basic inputs
let SERVER_FEATURES = [];
async function loadFeatures(){
  const res = await doFetch('/features');
  if(res && res.features){
    SERVER_FEATURES = res.features;
    renderFeatureInputs(SERVER_FEATURES);
  }
}

function renderFeatureInputs(features){
  const container = document.getElementById('features-list');
  container.innerHTML = '';
  const N = Math.min(features.length, 30);
  for(let i=0;i<N;i++){
    const name = features[i];
    const row = document.createElement('div');
    row.className = 'feature-row';
    const lbl = document.createElement('label'); lbl.className='small'; lbl.textContent = name;
    const inp = document.createElement('input'); inp.type='number'; inp.dataset.feat=name; inp.style.width='120px';
    row.appendChild(lbl); row.appendChild(inp);
    container.appendChild(row);
  }
  if(features.length > 30){
    const more = document.createElement('div'); more.className='muted'; more.textContent = `+ ${features.length-30} hidden features (use JSON editor to edit all)`;
    container.appendChild(more);
  }
}

document.getElementById('btn-fill-defaults').onclick = ()=>{
  // create a default empty object with zeros for visible features
  const obj = {};
  document.querySelectorAll('#features-list input[data-feat]').forEach(inp=>{ obj[inp.dataset.feat]=0.0; inp.value = 0.0 });
  document.getElementById('features-json').value = JSON.stringify(obj, null, 2);
}

document.getElementById('btn-load-json').onclick = async ()=>{
  const fileInput = document.getElementById('features-file');
  if(fileInput.files.length === 0){ alert('Choose a JSON file first'); return }
  const f = fileInput.files[0];
  const txt = await f.text();
  try{
    const obj = JSON.parse(txt);
    document.getElementById('features-json').value = JSON.stringify(obj, null, 2);
    // populate visible form inputs if keys match
    document.querySelectorAll('#features-list input[data-feat]').forEach(inp=>{ const key=inp.dataset.feat; if(key in obj) inp.value = obj[key]; });
  }catch(e){ alert('Invalid JSON: '+e); }
}

// When posting manual features, prefer JSON editor if non-empty, otherwise collect form inputs
async function buildFeaturesFromUI(){
  const txt = (document.getElementById('features-json').value || '').trim();
  if(txt){ try{ return JSON.parse(txt); }catch(e){ alert('Invalid JSON in editor'); return null } }
  const inputs = document.querySelectorAll('#features-list input[data-feat]');
  const obj = {};
  inputs.forEach(inp=>{ const k=inp.dataset.feat; if(inp.value !== '') obj[k]=parseFloat(inp.value); });
  return obj;
}

// Enhanced predict which supports manual features via editor or form
document.getElementById('btn-predict').onclick = async ()=>{
  const useLast = document.getElementById('use-last').checked;
  if(useLast){
    const out = await doFetch('/predict', { method:'POST', body: JSON.stringify({use_last_row:true})});
    document.getElementById('out-predict').textContent = JSON.stringify(out, null, 2);
    return;
  }
  const features = await buildFeaturesFromUI();
  if(features === null) return; // parsing error
  if(Object.keys(features).length === 0){ alert('No features provided; check JSON editor or form'); return }
  const out = await doFetch('/predict', { method:'POST', body: JSON.stringify({use_last_row:false, features: features})});
  document.getElementById('out-predict').textContent = JSON.stringify(out, null, 2);
}

document.getElementById('btn-ph').onclick = async ()=>{
  const H = parseInt(document.getElementById('horizon').value || 3, 10);
  const hold = document.getElementById('hold-exog').checked;
  const body = { horizon: H, hold_exog: hold };
  const out = await doFetch('/predict_horizon', { method:'POST', body: JSON.stringify(body) });
  document.getElementById('out-ph').textContent = JSON.stringify(out, null, 2);
}

// Optionally call health on load
document.getElementById('btn-health').click();
loadFeatures();
