// Configuração do gráfico
const glucoseChart = new Plotly.newPlot('glucose-chart', [{
    y: [],
    type: 'scatter',
    name: 'Glicemia',
    line: { color: '#2ecc71' }
}], {
    title: 'Monitoramento de Glicemia',
    yaxis: {
        title: 'mg/dL',
        range: [40, 400],
        gridcolor: '#eee'
    },
    xaxis: { title: 'Tempo' },
    annotations: [{
        y: 180,
        line: { color: '#e74c3c', dash: 'dash' }
    }, {
        y: 70,
        line: { color: '#e74c3c', dash: 'dash' }
    }]
});

// WebSocket para dados em tempo real
const ws = new WebSocket('ws://localhost:8000/ws');
let glucoseData = [];
let insulinData = [];

ws.onmessage = function (event) {
    const data = JSON.parse(event.data);

    if (data.type === 'glucose') {
        updateGlucoseChart(data.value);
        updateMetrics(data);
    }
};

function updateGlucoseChart(value) {
    glucoseData.push(value);
    if (glucoseData.length > 100) glucoseData.shift();

    Plotly.update('glucose-chart', {
        y: [glucoseData]
    });
}

function updateMetrics(data) {
    document.getElementById('tir').textContent =
        `Tempo em faixa: ${data.metrics.tir.toFixed(1)}%`;
    document.getElementById('hypo-risk').textContent =
        `Risco de hipo: ${data.metrics.hypo_risk.toFixed(1)}%`;
    document.getElementById('variability').textContent =
        `Variabilidade: ${data.metrics.cv.toFixed(1)}%`;
}

// Funções do Modal de Refeição
function openMealModal() {
    document.getElementById('mealModal').style.display = 'block';
}

function closeMealModal() {
    document.getElementById('mealModal').style.display = 'none';
}

// Iniciar sistema automaticamente
async function startSystem() {
    try {
        const response = await fetch('/api/control', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                action: 'start',
                parameters: {
                    duration_hours: 24,
                    acceleration: 1
                }
            })
        });

        if (!response.ok) throw new Error('Erro ao iniciar sistema');

        const result = await response.json();
        console.log('Sistema iniciado:', result);

    } catch (error) {
        console.error('Erro:', error);
        alert('Erro ao iniciar sistema');
    }
}

// Chamar ao carregar a página
document.addEventListener('DOMContentLoaded', startSystem);

async function addMeal(event) {
    event.preventDefault();

    try {
        // Verificar status do sistema
        const statusResponse = await fetch('/api/status');
        const status = await statusResponse.json();

        if (!status.is_running) {
            alert('Sistema não está rodando. Iniciando...');
            await startSystem();
        }

        const mealData = {
            carbs: parseFloat(document.getElementById('carbsInput').value),
            absorption_profile: document.getElementById('absorptionProfile').value,
            gi_index: parseFloat(document.getElementById('giIndex').value)
        };

        const response = await fetch('/api/meal', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(mealData)
        });

        if (!response.ok) throw new Error('Erro ao adicionar refeição');

        const result = await response.json();
        console.log('Refeição adicionada:', result);

        closeMealModal();
        document.getElementById('mealForm').reset();

    } catch (error) {
        console.error('Erro:', error);
        alert('Erro ao adicionar refeição: ' + error.message);
    }
}
