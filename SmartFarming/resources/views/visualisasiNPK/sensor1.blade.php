<div class="row">
    <div class="col-12 col-lg-6">
        <div class="card" style="border-color: #CED4DA">
            <div class="card-body">
                <h6 id="titleTemperature_sensor2" data-original="Suhu Tanah">Suhu Tanah</h6>
                <div class="ratio ratio-16x9">
                    <iframe id="grafanaIframeTemperature_sensor2"
                        src="http://labai.polinema.ac.id:3010/d-solo/eempvyqjk5csgf/website-visualisasi-data?orgId=1&timezone=browser&theme=light&panelId=12&__feature.dashboardSceneSolo"
                        allowfullscreen style="display: none"></iframe>
                </div>
            </div>
        </div>
    </div>
    <div class="col-12 col-lg-6">
        <div class="card" style="border-color: #CED4DA">
            <div class="card-body">
                <h6 id="titleHumidity_sensor2" data-original="Kelembapan Tanah">Kelembapan Tanah</h6>
                <div class="ratio ratio-16x9">
                    <iframe id="grafanaIframeHumidity_sensor2"
                        src="http://labai.polinema.ac.id:3010/d-solo/eempvyqjk5csgf/website-visualisasi-data?orgId=1&timezone=browser&theme=light&panelId=13&__feature.dashboardSceneSolo"
                        allowfullscreen style="display: none"></iframe>
                </div>
            </div>
        </div>
    </div>
</div>
<div class="row">
    <div class="col-12 col-lg-6">
        <div class="card" style="border-color: #CED4DA">
            <div class="card-body">
                <h6 id="titleConductivity_sensor2" data-original="Konduktivitas Tanah">Konduktivitas Tanah</h6>
                <div class="ratio ratio-16x9">
                    <iframe id="grafanaIframeConductivity_sensor2"
                        src="http://labai.polinema.ac.id:3010/d-solo/eempvyqjk5csgf/website-visualisasi-data?orgId=1&timezone=browser&theme=light&panelId=14&__feature.dashboardSceneSolo"
                        allowfullscreen style="display: none"></iframe>
                </div>
            </div>
        </div>
    </div>
    <div class="col-12 col-lg-6">
        <div class="card" style="border-color: #CED4DA">
            <div class="card-body">
                <h6 id="titlepH_sensor2" data-original="pH Tanah">pH Tanah</h6>
                <div class="ratio ratio-16x9">
                    <iframe id="grafanaIframePh_sensor2"
                        src="http://labai.polinema.ac.id:3010/d-solo/eempvyqjk5csgf/website-visualisasi-data?orgId=1&timezone=browser&theme=light&panelId=15&__feature.dashboardSceneSolo"
                        allowfullscreen style="display: none"></iframe>
                </div>
            </div>
        </div>
    </div>
</div>
<div class="row">
    <div class="col-12 col-lg-6">
        <div class="card" style="border-color: #CED4DA">
            <div class="card-body">
                <h6 id="titleNitrogen_sensor2" data-original="Nitrogen Tanah">Nitrogen Tanah</h6>
                <div class="ratio ratio-16x9">
                    <iframe id="grafanaIframeNitrogen_sensor2"
                        src="http://labai.polinema.ac.id:3010/d-solo/eempvyqjk5csgf/website-visualisasi-data?orgId=1&timezone=browser&theme=light&panelId=16&__feature.dashboardSceneSolo"
                        allowfullscreen style="display: none"></iframe>
                </div>
            </div>
        </div>
    </div>
    <div class="col-12 col-lg-6">
        <div class="card" style="border-color: #CED4DA">
            <div class="card-body">
                <h6 id="titlePhosphorus_sensor2" data-original="Fosfor Tanah">Fosfor Tanah</h6>
                <div class="ratio ratio-16x9">
                    <iframe id="grafanaIframePhosphorus_sensor2"
                        src="http://labai.polinema.ac.id:3010/d-solo/eempvyqjk5csgf/website-visualisasi-data?orgId=1&timezone=browser&theme=light&panelId=18&__feature.dashboardSceneSolo"
                        allowfullscreen style="display: none"></iframe>
                </div>
            </div>
        </div>
    </div>
</div>
<div class="row">
    <div class="col-12 col-lg-6">
        <div class="card" style="border-color: #CED4DA">
            <div class="card-body">
                <h6 id="titlePotassium_sensor2" data-original="Kalium Tanah">Kalium Tanah</h6>
                <div class="ratio ratio-16x9">
                    <iframe id="grafanaIframePotassium_sensor2"
                        src="http://labai.polinema.ac.id:3010/d-solo/eempvyqjk5csgf/website-visualisasi-data?orgId=1&timezone=browser&theme=light&panelId=17&__feature.dashboardSceneSolo"
                        allowfullscreen style="display: none"></iframe>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
    const iframeIds = [
        'grafanaIframeTemperature_sensor2',
        'grafanaIframeHumidity_sensor2',
        'grafanaIframeConductivity_sensor2',
        'grafanaIframePh_sensor2',
        'grafanaIframeNitrogen_sensor2',
        'grafanaIframePhosphorus_sensor2',
        'grafanaIframePotassium_sensor2'
    ];

    let loadedCount = 0;
    const totalIframes = iframeIds.length;

    function showAllIframes() {
        iframeIds.forEach(id => {
            const iframe = document.getElementById(id);
            if (iframe) iframe.style.display = 'block';
        });
    }

    iframeIds.forEach(id => {
        const iframe = document.getElementById(id);
        if (iframe) {
            iframe.onload = function() {
                loadedCount++;
                if (loadedCount === totalIframes) {
                    showAllIframes();
                }
            };
        }
    });
</script>
