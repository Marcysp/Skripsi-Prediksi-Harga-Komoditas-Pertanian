<div class="row">
    <div class="col-12 col-lg-6">
        <div class="card" style="border-color: #CED4DA">
            <div class="card-body">
                <h6 id="titleTemperature_sensor3" data-original="Suhu Tanah">Suhu Tanah</h6>
                <div class="ratio ratio-16x9">
                    <iframe id="grafanaIframeTemperature_sensor3"
                        src="http://labai.polinema.ac.id:3010/d-solo/eempvyqjk5csgf/website-visualisasi-data?orgId=1&timezone=browser&theme=light&panelId=19&__feature.dashboardSceneSolo"
                        allowfullscreen style="display: none"></iframe>
                </div>
            </div>
        </div>
    </div>
    <div class="col-12 col-lg-6">
        <div class="card" style="border-color: #CED4DA">
            <div class="card-body">
                <h6 id="titleHumidity_sensor3" data-original="Kelembapan Tanah">Kelembapan Tanah</h6>
                <div class="ratio ratio-16x9">
                    <iframe id="grafanaIframeHumidity_sensor3"
                        src="http://labai.polinema.ac.id:3010/d-solo/eempvyqjk5csgf/website-visualisasi-data?orgId=1&timezone=browser&theme=light&panelId=20&__feature.dashboardSceneSolo"
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
                <h6 id="titleConductivity_sensor3" data-original="Konduktivitas Tanah">Konduktivitas Tanah</h6>
                <div class="ratio ratio-16x9">
                    <iframe id="grafanaIframeConductivity_sensor3"
                        src="http://labai.polinema.ac.id:3010/d-solo/eempvyqjk5csgf/website-visualisasi-data?orgId=1&timezone=browser&theme=light&panelId=21&__feature.dashboardSceneSolo"
                        allowfullscreen style="display: none"></iframe>
                </div>
            </div>
        </div>
    </div>
    <div class="col-12 col-lg-6">
        <div class="card" style="border-color: #CED4DA">
            <div class="card-body">
                <h6 id="titlepH_sensor3" data-original="pH Tanah">pH Tanah</h6>
                <div class="ratio ratio-16x9">
                    <iframe id="grafanaIframePh_sensor3"
                        src="http://labai.polinema.ac.id:3010/d-solo/eempvyqjk5csgf/website-visualisasi-data?orgId=1&timezone=browser&theme=light&panelId=22&__feature.dashboardSceneSolo"
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
                <h6 id="titleNitrogen_sensor3" data-original="Nitrogen Tanah">Nitrogen Tanah</h6>
                <div class="ratio ratio-16x9">
                    <iframe id="grafanaIframeNitrogen_sensor3"
                        src="http://labai.polinema.ac.id:3010/d-solo/eempvyqjk5csgf/website-visualisasi-data?orgId=1&timezone=browser&theme=light&panelId=23&__feature.dashboardSceneSolo"
                        allowfullscreen style="display: none"></iframe>
                </div>
            </div>
        </div>
    </div>
    <div class="col-12 col-lg-6">
        <div class="card" style="border-color: #CED4DA">
            <div class="card-body">
                <h6 id="titlePhosphorus_sensor3" data-original="Fosfor Tanah">Fosfor Tanah</h6>
                <div class="ratio ratio-16x9">
                    <iframe id="grafanaIframePhosphorus_sensor3"
                        src="http://labai.polinema.ac.id:3010/d-solo/eempvyqjk5csgf/website-visualisasi-data?orgId=1&timezone=browser&theme=light&panelId=24&__feature.dashboardSceneSolo"
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
                <h6 id="titlePotassium_sensor3" data-original="Kalium Tanah">Soil Potassium</h6>
                <div class="ratio ratio-16x9">
                    <iframe id="grafanaIframePotassium_sensor3"
                        src="http://labai.polinema.ac.id:3010/d-solo/eempvyqjk5csgf/website-visualisasi-data?orgId=1&timezone=browser&theme=light&panelId=25&__feature.dashboardSceneSolo"
                        allowfullscreen style="display: none"></iframe>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
    const iframeIds = [
        'grafanaIframeTemperature_sensor3',
        'grafanaIframeHumidity_sensor3',
        'grafanaIframeConductivity_sensor3',
        'grafanaIframePh_sensor3',
        'grafanaIframeNitrogen_sensor3',
        'grafanaIframePhosphorus_sensor3',
        'grafanaIframePotassium_sensor3'
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
