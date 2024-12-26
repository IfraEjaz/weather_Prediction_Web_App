// Historical and Forecast Data Plotting
const forecastCtx = document.getElementById('forecastChart').getContext('2d');

// Forecast Data from Django Context
const forecastTemps = [
    parseFloat('{{ temp1 }}'),
    parseFloat('{{ temp2 }}'),
    parseFloat('{{ temp3 }}'),
    parseFloat('{{ temp4 }}'),
    parseFloat('{{ temp5 }}'),
];
const forecastTimes = ['{{ time1 }}', '{{ time2 }}', '{{ time3 }}', '{{ time4 }}', '{{ time5 }}'];

// Historical Data from Django Context
const historicalTemps = JSON.parse('{{ historical_data|safe }}').map(data => data.temp);
const historicalTimes = JSON.parse('{{ historical_data|safe }}').map(data => data.date);

// Combine Historical and Forecast Data
const allTemps = historicalTemps.concat(forecastTemps);
const allTimes = historicalTimes.concat(forecastTimes);

new Chart(forecastCtx, {
    type: 'line',
    data: {
        labels: allTimes,
        datasets: [{
            label: 'Temperature (°C)',
            data: allTemps,
            borderColor: 'blue',
            fill: false,
        }]
    },
    options: {
        responsive: true,
        plugins: {
            legend: { display: true },
            title: { display: true, text: 'Temperature Trends (Past & Future)' }
        }
    }
});
