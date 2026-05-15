// UI Elements
const dimSlider = document.getElementById('dim-slider');
const dimVal = document.getElementById('dim-val');
const probSlider = document.getElementById('prob-slider');
const probVal = document.getElementById('prob-val');
const epochsSlider = document.getElementById('epochs-slider');
const epochsVal = document.getElementById('epochs-val');
const energySlider = document.getElementById('energy-slider');
const energyVal = document.getElementById('energy-val');

const btnStart = document.getElementById('start-btn');
const btnStop = document.getElementById('stop-btn');
const btnPysr = document.getElementById('pysr-btn');

const displayIc = document.getElementById('ic-display');
const displayNpt = document.getElementById('npt-display');
const displayTp = document.getElementById('tp-display');
const displayEpoch = document.getElementById('epoch-display');
const consoleOutput = document.getElementById('console-output');

// Chart Setup
const ctx = document.getElementById('icChart').getContext('2d');
const gradient = ctx.createLinearGradient(0, 0, 0, 300);
gradient.addColorStop(0, 'rgba(0, 240, 255, 0.5)');
gradient.addColorStop(1, 'rgba(0, 240, 255, 0.0)');

const chart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Coherent Information (I_c)',
            data: [],
            borderColor: '#00f0ff',
            backgroundColor: gradient,
            borderWidth: 2,
            pointRadius: 0,
            fill: true,
            tension: 0.4
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        scales: {
            x: { 
                grid: { color: 'rgba(255,255,255,0.05)' },
                ticks: { color: '#8a8aC0' }
            },
            y: { 
                grid: { color: 'rgba(255,255,255,0.05)' },
                ticks: { color: '#8a8aC0' }
            }
        },
        plugins: {
            legend: { display: false }
        }
    }
});

// Update slider values
dimSlider.addEventListener('input', () => dimVal.innerText = dimSlider.value);
probSlider.addEventListener('input', () => probVal.innerText = probSlider.value);
epochsSlider.addEventListener('input', () => epochsVal.innerText = epochsSlider.value);
energySlider.addEventListener('input', () => {
    energyVal.innerText = parseFloat(energySlider.value) === 15.0 ? 'MAX' : energySlider.value;
});

// Reactive UI Logic
const topologySelect = document.getElementById('topology-select');
const topologyVal = document.getElementById('topology-val');
const noiseSelect = document.getElementById('noise-select');
const noiseVal = document.getElementById('noise-val');
const objectiveSelect = document.getElementById('objective-select');
const objectiveVal = document.getElementById('objective-val');
const ancillaToggle = document.getElementById('ancilla-toggle');

function applyConstraints() {
    // 1. Update text labels to match dropdowns
    noiseVal.innerText = noiseSelect.options[noiseSelect.selectedIndex].text.split(' ')[0];
    topologyVal.innerText = topologySelect.options[topologySelect.selectedIndex].text.split(' ')[0];
    objectiveVal.innerText = objectiveSelect.options[objectiveSelect.selectedIndex].text.split(' ')[0];

    // 2. Topology Constraints (Protect GPU)
    if (topologySelect.value === 'tripartite') {
        dimSlider.max = 5;
        if (parseInt(dimSlider.value) > 5) {
            dimSlider.value = 4;
            dimVal.innerText = 4;
        }
    } else {
        dimSlider.max = 16;
    }

    // 3. Noise Model Constraints
    // Removed to give user full freedom over Deep Quantum Search
    ancillaToggle.disabled = false;
    ancillaToggle.parentElement.style.opacity = '1';

    // 4. Objective Constraints
    if (objectiveSelect.value === 'metrology') {
        chart.data.datasets[0].label = 'Sensitivity (Trace Dist)';
        chart.data.datasets[0].borderColor = '#ff00ff'; // Magenta
    } else {
        chart.data.datasets[0].label = 'Coherent Information (I_c)';
        chart.data.datasets[0].borderColor = '#00f0ff'; // Cyan
    }
    chart.update();
}

// Attach event listeners
topologySelect.addEventListener('change', () => {
    applyConstraints();
    if (topologySelect.value === 'tripartite') log('Tripartite topology selected. Dimension locked to max 5.', 'warn');
});

noiseSelect.addEventListener('change', () => {
    applyConstraints();
    if (noiseSelect.value !== 'erasure') log('Deep Quantum Search locked to ON for non-Erasure models.', 'info');
});

objectiveSelect.addEventListener('change', () => {
    applyConstraints();
    log(`Objective switched to ${objectiveSelect.value}.`, 'info');
});

// Run once on load to set initial state
applyConstraints();

function log(msg, type='normal') {
    const p = document.createElement('p');
    p.innerText = `> ${msg}`;
    if (type !== 'normal') p.classList.add(type);
    consoleOutput.appendChild(p);
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

// WebSocket logic
let ws = null;

btnPysr.addEventListener('click', async () => {
    log('Initiating PySR Symbolic Extraction from saved matrices...', 'info');
    btnPysr.disabled = true;
    try {
        const response = await fetch('/extract_formula');
        const data = await response.json();
        if(data.status === 'success') {
            log('PySR Extraction complete! See terminal for algebraic formula.', 'success');
        } else {
            log('PySR Error: ' + data.message, 'error');
        }
    } catch (e) {
        log('Error contacting server for PySR.', 'error');
    }
    btnPysr.disabled = false;
});

btnStart.addEventListener('click', () => {
    if (ws) ws.close();

    // Reset UI
    chart.data.labels = [];
    chart.data.datasets[0].data = [];
    chart.update();
    consoleOutput.innerHTML = '';
    
    btnStart.disabled = true;
    btnStop.disabled = false;
    
    dimSlider.disabled = true;
    probSlider.disabled = true;
    epochsSlider.disabled = true;

    // Connect to WebSocket
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws/synthesize`);

    ws.onopen = () => {
        log('Connection established. Sending parameters...', 'info');
        ws.send(JSON.stringify({
            noise: document.getElementById('noise-select').value,
            topology: document.getElementById('topology-select').value,
            objective: document.getElementById('objective-select').value,
            d: dimSlider.value,
            energy: energySlider.value,
            p: probSlider.value,
            epochs: epochsSlider.value,
            ancilla: document.getElementById('ancilla-toggle').checked
        }));
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'log') {
            log(data.message);
        } 
        else if (data.type === 'update') {
            // Update Text
            displayIc.innerText = data.ic > 0 ? `+${data.ic.toFixed(5)}` : data.ic.toFixed(5);
            displayNpt.innerText = data.npt.toExponential(2);
            displayTp.innerText = data.tp.toExponential(2);
            displayEpoch.innerText = `${data.epoch} / ${epochsSlider.value}`;

            if (data.ic > 0) {
                displayIc.style.color = '#00ff00';
                displayIc.style.textShadow = '0 0 10px rgba(0,255,0,0.5)';
            } else {
                displayIc.style.color = 'var(--accent-magenta)';
                displayIc.style.textShadow = '0 0 10px rgba(255,0,255,0.4)';
            }

            // Update Chart
            chart.data.labels.push(data.epoch);
            chart.data.datasets[0].data.push(data.ic);
            chart.update();
            
            if (data.is_best) {
                log(`New Best I_c: ${data.ic.toFixed(6)}`, 'info');
            }
        }
        else if (data.type === 'done') {
            log(`Synthesis Complete. Best I_c: ${data.best_ic.toFixed(5)}`, 'info');
            if (data.best_ic > 0) {
                log('SUPERACTIVATION ACHIEVED! 🎉', 'info');
            } else {
                log('Channel capacity converges to 0. Try lowering p.', 'warn');
            }
            stopEngine();
        }
        else if (data.type === 'error') {
            log(`ERROR: ${data.message}`, 'err');
            stopEngine();
        }
    };

    ws.onclose = () => {
        stopEngine();
    };
});

btnStop.addEventListener('click', () => {
    if (ws) {
        ws.close();
        log('Engine halted by user.', 'warn');
    }
    stopEngine();
});

function stopEngine() {
    btnStart.disabled = false;
    btnStop.disabled = true;
    dimSlider.disabled = false;
    probSlider.disabled = false;
    epochsSlider.disabled = false;
}
