@extends('layout.template')
@section('title', 'Monitoring Cuaca | Agrilink Vocpro')
@section('content')
    <div class="container-fluid">
        <div class="row">
            <div class="form-group col-12 col-lg-3 ms-auto">
                <label for="start_date" class="form-label">Filter Tanggal:</label>
                <input type="text" class="form-control" name="daterange" id="daterange" placeholder="Masukkan tanggal">
            </div>
        </div>
        <div class="row">
            @foreach ($weatherData->slice(0, 4) as $item)
                <div class="col-12 col-lg-3">
                    <div class="card text-center p-3" style="border-color: #CED4DA">
                        <div class="mb-2">
                            <i class="{{ $item['icon'] }} fs-1" style="color: #227066"></i>
                        </div>
                        <div class="fw-bold">{{ $item['label'] }}</div>
                        <div class="fs-4">{{ $item['value'] }} {{ $item['unit'] }}</div>
                    </div>
                </div>
            @endforeach
        </div>
        <div class="row">
            <div class="col-12 col-lg-6">
                <div class="card" style="border-color: #CED4DA">
                    <div class="card-body">
                        <h6 id="titleTemperature" data-original="Suhu">Suhu</h6>
                        <div class="ratio ratio-16x9">
                            <iframe id="grafanaIframeTemperature"
                                src="http://labai.polinema.ac.id:3010/d-solo/8e8fc548-1ffd-4056-bc61-b38357d2d30a/cuaca-terkini?orgId=1&timezone=browser&refresh=1h&theme=light&panelId=1&__feature.dashboardSceneSolo"
                                allowfullscreen style="display: none"></iframe>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-12 col-lg-6">
                <div class="card" style="border-color: #CED4DA">
                    <div class="card-body">
                        <h6 id="titleCloudCover" data-original="Tutupan Awan">Tutupan Awan</h6>
                        <div class="ratio ratio-16x9">
                            <iframe id="grafanaIframeCloudCover"
                                src="http://labai.polinema.ac.id:3010/d-solo/8e8fc548-1ffd-4056-bc61-b38357d2d30a/cuaca-terkini?orgId=1&timezone=browser&refresh=1h&theme=light&panelId=3&__feature.dashboardSceneSolo"
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
                        <h6 id="titleWindSpeed" data-original="Kecepatan Angin">Kecepatan Angin</h6>
                        <div class="ratio ratio-16x9">
                            <iframe id="grafanaIframeWindSpeed"
                                src="http://labai.polinema.ac.id:3010/d-solo/8e8fc548-1ffd-4056-bc61-b38357d2d30a/cuaca-terkini?orgId=1&timezone=browser&refresh=1h&theme=light&panelId=4&__feature.dashboardSceneSolo"
                                allowfullscreen style="display: none"></iframe>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-12 col-lg-6">
                <div class="card" style="border-color: #CED4DA">
                    <div class="card-body">
                        <h6 id="titleWeather" data-original="Cuaca">Cuaca</h6>
                        <div class="ratio ratio-16x9">
                            <iframe id="grafanaIframeWeather"
                                src="http://labai.polinema.ac.id:3010/d-solo/8e8fc548-1ffd-4056-bc61-b38357d2d30a/cuaca-terkini?orgId=1&timezone=browser&refresh=1h&theme=light&panelId=5&__feature.dashboardSceneSolo"
                                allowfullscreen style="display: none"></iframe>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
@endsection
@push('css')
@endpush
@push('js')
    <script>
        const iframeIds = [
            'grafanaIframeTemperature',
            'grafanaIframeCloudCover',
            'grafanaIframeWindSpeed',
            'grafanaIframeWeather'
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

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const urlParams = new URLSearchParams(window.location.search);
            const startDateParam = urlParams.get('start_date');
            const endDateParam = urlParams.get('end_date');

            if (startDateParam && endDateParam) {
                const start = moment(startDateParam, "YYYY-MM-DD");
                const end = moment(endDateParam, "YYYY-MM-DD");

                $('#daterange').data('daterangepicker').setStartDate(start);
                $('#daterange').data('daterangepicker').setEndDate(end);
                $('#daterange').val(start.format('DD-MM-YY') + ' → ' + end.format('DD-MM-YY'));

                // Ini penting agar iframe langsung update
                setTimeout(() => {
                    updateGrafanaIframe(startDateParam, endDateParam);
                }, 1000); // tambahkan delay kecil untuk memastikan iframe ada di DOM
            } else {
                defaultGrafanaIframe();
            }
        });
    </script>

    <script>
        $('#daterange').daterangepicker({
            opens: 'left',
            autoUpdateInput: false,
            locale: {
                applyLabel: 'Pilih',
                cancelLabel: 'Batal',
                format: 'DD-MM-YYYY',
                daysOfWeek: ['Min', 'Sen', 'Sel', 'Rab', 'Kam', 'Jum', 'Sab'],
                monthNames: ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
                    'Juli', 'Agustus', 'September', 'Oktober', 'November',
                    'Desember'
                ],
            }
        });

        $('#daterange').on('apply.daterangepicker', function(ev, picker) {
            const startDate = picker.startDate.format('YYYY-MM-DD');
            const endDate = picker.endDate.format('YYYY-MM-DD');
            const today = moment().format('YYYY-MM-DD');

            if (endDate > today) {
                Swal.fire({
                    toast: true,
                    position: 'top-end',
                    icon: 'error',
                    title: 'Tanggal tidak boleh lebih dari hari ini',
                    showConfirmButton: false,
                    timer: 3000,
                    timerProgressBar: true,
                });
                $(this).val(''); // Reset input
                return;
            }

            $(this).val(picker.startDate.format('DD-MM-YY') + ' → ' + picker.endDate.format('DD-MM-YY'));
            const currentUrl = new URL(window.location.href);
            currentUrl.searchParams.set('start_date', startDate);
            currentUrl.searchParams.set('end_date', endDate);

            window.history.replaceState({}, '', currentUrl); // Update URL tanpa reload

            updateGrafanaIframe(startDate, endDate);
        });

        $('#daterange').on('cancel.daterangepicker', function(ev, picker) {
            $(this).val('');
            const currentUrl = new URL(window.location.href);
            currentUrl.searchParams.delete('start_date');
            currentUrl.searchParams.delete('end_date');
            window.history.replaceState({}, '', currentUrl); // Update URL tanpa reload
            defaultGrafanaIframe();
        });
    </script>

    <script>
        function updateGrafanaIframe(startDate, endDate) {
            const fromTimestamp = new Date(startDate).getTime();

            const toDate = new Date(endDate);
            toDate.setHours(23, 59, 59, 999); // pastikan akhir hari
            const toTimestamp = toDate.getTime();
            const baseGrafanaUrl =
                "http://labai.polinema.ac.id:3010/d-solo/8e8fc548-1ffd-4056-bc61-b38357d2d30a/cuaca-terkini?orgId=1";
            const commonParams =
                `&from=${fromTimestamp}&to=${toTimestamp}&timezone=Asia%2FJakarta&refresh=1h&theme=light&__feature.dashboardSceneSolo`;

            const panels = [{
                    id: 1,
                    elementId: "grafanaIframeTemperature"
                },
                {
                    id: 3,
                    elementId: "grafanaIframeCloudCover"
                },
                {
                    id: 4,
                    elementId: "grafanaIframeWindSpeed"
                },
                {
                    id: 5,
                    elementId: "grafanaIframeWeather"
                }
            ];

            panels.forEach(panel => {
                const url = `${baseGrafanaUrl}${commonParams}&panelId=${panel.id}`;
                const iframe = document.getElementById(panel.elementId);
                if (iframe) {
                    iframe.src = url;
                }
            });
        }
    </script>
    <script>
        function defaultGrafanaIframe() {
            const baseGrafanaUrl =
                "http://labai.polinema.ac.id:3010/d-solo/8e8fc548-1ffd-4056-bc61-b38357d2d30a/cuaca-terkini?orgId=1";
            const commonParams = "&timezone=browser&refresh=1h&theme=light&__feature.dashboardSceneSolo"

            const panels = [{
                    id: 1,
                    elementId: "grafanaIframeTemperature"
                },
                {
                    id: 3,
                    elementId: "grafanaIframeCloudCover"
                },
                {
                    id: 4,
                    elementId: "grafanaIframeWindSpeed"
                },
                {
                    id: 5,
                    elementId: "grafanaIframeWeather"
                }
            ];

            panels.forEach(panel => {
                const iframe = document.getElementById(panel.elementId);
                if (iframe) {
                    iframe.src = `${baseGrafanaUrl}${commonParams}&panelId=${panel.id}`;
                }
            });
        }
    </script>
@endpush
