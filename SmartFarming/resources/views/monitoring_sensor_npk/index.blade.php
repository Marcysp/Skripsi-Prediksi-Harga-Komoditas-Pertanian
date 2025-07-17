@extends('layout.template')
@section('title', 'Monitoring NPK | Agrilink Vocpro')
@section('content')
    <div class="container-fluid">
        <div class="d-flex align-items-end form-group row mb-0">
            <div class="form-group col-12 col-lg-3 mb-0">
                <label for="sensor_npk" class="form-label">Sensor NPK:</label>
                <div class="form-group">
                    <select class="choices form-select" name="selected_sensor_npk" id="selected_sensor_npk" required>
                        <option value="">- Semua Sensor -</option>
                        @foreach ($sensor_npk as $item)
                            <option value="{{ $item->id }}"
                                {{ request()->get('selected_sensor_npk') == $item->id ? 'selected' : '' }}>
                                {{ $item->public_name }}
                            </option>
                        @endforeach
                    </select>
                </div>
            </div>
            <div class="form-group col-12 col-lg-3 ms-auto">
                <label for="start_date" class="form-label">Filter Tanggal:</label>
                <input type="text" class="form-control" name="daterange" id="daterange" placeholder="Masukkan tanggal">
            </div>
        </div>
        <!-- Carousel Mulai dari sini -->
        <div id="carouselNPK" class="carousel slide" data-bs-ride="carousel">
            <div class="carousel-inner">
                {{-- Slide Pertama --}}
                <div class="carousel-item active">
                    <div class="row">
                        @foreach ($dataNPK->slice(0, 4) as $item)
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
                </div>

                {{-- Slide Kedua --}}
                <div class="carousel-item">
                    <div class="row">
                        @foreach ($dataNPK->slice(4, 4) as $item)
                            <div class="col-md-3">
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
                </div>
            </div>
            <!-- Navigasi -->
            <button class="carousel-control-prev" type="button" data-bs-target="#carouselNPK" data-bs-slide="prev">
                <span class="carousel-control-prev-icon"></span>
            </button>
            <button class="carousel-control-next" type="button" data-bs-target="#carouselNPK" data-bs-slide="next">
                <span class="carousel-control-next-icon"></span>
            </button>
        </div>
        <div id="visualisasi_data">
            @php
                $selectedSensor = request()->get('selected_sensor_npk');
            @endphp

            @if ($selectedSensor == '2')
                @include('visualisasiNPK.sensor1');
            @elseif ($selectedSensor == '3')
                @include('visualisasiNPK.sensor2')
            @else
                @include('visualisasiNPK.sensorAll')
            @endif
        </div>
    </div>
@endsection
@push('css')
@endpush
@push('js')
    <script>
        const npkFilter = document.getElementById('selected_sensor_npk');

        npkFilter.addEventListener('change', function() {
            const selectedValue = this.value;
            const currentUrl = new URL(window.location.href);
            currentUrl.searchParams.set('selected_sensor_npk', selectedValue);
            window.location.href = currentUrl.toString(); // Redirect ke URL baru dengan param
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
            const selectedSensor = '{{ request()->get('selected_sensor_npk') }}' || 'all'; // sensor ID atau all

            const panelMapSensor2 = {
                'Temperature': 12,
                'Humidity': 13,
                'Conductivity': 14,
                'Ph': 15,
                'Nitrogen': 16,
                'Phosphorus': 17,
                'Potassium': 18,
            };

            const panelMapSensor3 = {
                'Temperature': 19,
                'Humidity': 20,
                'Conductivity': 21,
                'Ph': 22,
                'Nitrogen': 23,
                'Phosphorus': 24,
                'Potassium': 25,
            };

            const panelMapSensorAll = {
                'Temperature': 26,
                'Humidity': 27,
                'Conductivity': 28,
                'Ph': 29,
                'Nitrogen': 30,
                'Phosphorus': 31,
                'Potassium': 32,
            };

            if (selectedSensor == 2) {
                Object.entries(panelMapSensor2).forEach(([label, panelId]) => {
                    const iframeId = `grafanaIframe${label}_sensor${selectedSensor}`;
                    const iframe = document.getElementById(iframeId);
                    if (!iframe) return;

                    const from = new Date(startDate).getTime();
                    const to = new Date(endDate).getTime();

                    const newSrc =
                        `http://labai.polinema.ac.id:3010/d-solo/aembuxu4ks5q8c/rata-rata-harian?orgId=1&from=${from}&to=${to}&timezone=browser&refresh=1d&theme=light&panelId=${panelId}&__feature=dashboardSceneSolo`;

                    iframe.src = newSrc;
                });

                const titleIframeIds = [
                    'titleTemperature_sensor2',
                    'titleHumidity_sensor2',
                    'titleConductivity_sensor2',
                    'titlepH_sensor2',
                    'titleNitrogen_sensor2',
                    'titlePhosphorus_sensor2',
                    'titlePotassium_sensor2',
                ];

                titleIframeIds.forEach(id => {
                    const el = document.getElementById(id);
                    if (el) {
                        const original = el.dataset.original;
                        el.innerText = 'Rata-Rata Harian ' + original;
                    }
                });
            } else if (selectedSensor == 3) {
                Object.entries(panelMapSensor3).forEach(([label, panelId]) => {
                    const iframeId = `grafanaIframe${label}_sensor${selectedSensor}`;
                    const iframe = document.getElementById(iframeId);
                    if (!iframe) return;

                    const from = new Date(startDate).getTime();
                    const to = new Date(endDate).getTime();

                    const newSrc =
                        `http://labai.polinema.ac.id:3010/d-solo/aembuxu4ks5q8c/rata-rata-harian?orgId=1&from=${from}&to=${to}&timezone=browser&refresh=1d&theme=light&panelId=${panelId}&__feature=dashboardSceneSolo`;

                    iframe.src = newSrc;
                });

                const titleIframeIds = [
                    'titleTemperature_sensor3',
                    'titleHumidity_sensor3',
                    'titleConductivity_sensor3',
                    'titlepH_sensor3',
                    'titleNitrogen_sensor3',
                    'titlePhosphorus_sensor3',
                    'titlePotassium_sensor3',
                ];

                titleIframeIds.forEach(id => {
                    const el = document.getElementById(id);
                    if (el) {
                        const original = el.dataset.original;
                        el.innerText = 'Rata-Rata Harian ' + original;
                    }
                });
            } else if (selectedSensor == 'all') {
                Object.entries(panelMapSensorAll).forEach(([label, panelId]) => {
                    const iframeId = `grafanaIframe${label}_sensorAll`;
                    const iframe = document.getElementById(iframeId);
                    if (!iframe) return;

                    const from = new Date(startDate).getTime();
                    const to = new Date(endDate).getTime();

                    const newSrc =
                        `http://labai.polinema.ac.id:3010/d-solo/aembuxu4ks5q8c/rata-rata-harian?orgId=1&from=${from}&to=${to}&timezone=browser&refresh=1d&theme=light&panelId=${panelId}&__feature=dashboardSceneSolo`;

                    iframe.src = newSrc;
                });

                const titleIframeIds = [
                    'titleTemperature_sensorAll',
                    'titleHumidity_sensorAll',
                    'titleConductivity_sensorAll',
                    'titlepH_sensorAll',
                    'titleNitrogen_sensorAll',
                    'titlePhosphorus_sensorAll',
                    'titlePotassium_sensorAll',
                ];

                titleIframeIds.forEach(id => {
                    const el = document.getElementById(id);
                    if (el) {
                        const original = el.dataset.original;
                        el.innerText = 'Rata-Rata Harian ' + original;
                    }
                });
            }
        }
    </script>

    <script>
        function defaultGrafanaIframe() {
            const selectedSensor = '{{ request()->get('selected_sensor_npk') }}' || 'all';

            const panelMapSensor2 = {
                'Temperature': 12,
                'Humidity': 13,
                'Conductivity': 14,
                'Ph': 15,
                'Nitrogen': 16,
                'Phosphorus': 17,
                'Potassium': 18,
            };

            const panelMapSensor3 = {
                'Temperature': 19,
                'Humidity': 20,
                'Conductivity': 21,
                'Ph': 22,
                'Nitrogen': 23,
                'Phosphorus': 24,
                'Potassium': 25,
            };

            const panelMapSensorAll = {
                'Temperature': 27,
                'Humidity': 28,
                'Conductivity': 29,
                'Ph': 30,
                'Nitrogen': 31,
                'Phosphorus': 32,
                'Potassium': 33,
            };

            if (selectedSensor == 2) {
                Object.entries(panelMapSensor2).forEach(([label, panelId]) => {
                    const iframeId = `grafanaIframe${label}_sensor${selectedSensor}`;
                    const iframe = document.getElementById(iframeId);
                    if (!iframe) return;

                    const defaultSrc =
                        `http://labai.polinema.ac.id:3010/d-solo/eempvyqjk5csgf/website-visualisasi-data?orgId=1&timezone=browser&theme=light&panelId=${panelId}&__feature=dashboardSceneSolo`;
                    iframe.src = defaultSrc;
                });

                const titleIframeIds = [
                    'titleTemperature_sensor2',
                    'titleHumidity_sensor2',
                    'titleConductivity_sensor2',
                    'titlepH_sensor2',
                    'titleNitrogen_sensor2',
                    'titlePhosphorus_sensor2',
                    'titlePotassium_sensor2',
                ];

                titleIframeIds.forEach(id => {
                    const el = document.getElementById(id);
                    if (el) {
                        el.innerText = el.dataset.original;
                    }
                });
            } else if (selectedSensor == 3) {
                Object.entries(panelMapSensor3).forEach(([label, panelId]) => {
                    const iframeId = `grafanaIframe${label}_sensor${selectedSensor}`;
                    const iframe = document.getElementById(iframeId);
                    if (!iframe) return;

                    const defaultSrc =
                        `http://labai.polinema.ac.id:3010/d-solo/eempvyqjk5csgf/website-visualisasi-data?orgId=1&timezone=browser&theme=light&panelId=${panelId}&__feature=dashboardSceneSolo`;
                    iframe.src = defaultSrc;
                });

                const titleIframeIds = [
                    'titleTemperature_sensor3',
                    'titleHumidity_sensor3',
                    'titleConductivity_sensor3',
                    'titlepH_sensor3',
                    'titleNitrogen_sensor3',
                    'titlePhosphorus_sensor3',
                    'titlePotassium_sensor3',
                ];

                titleIframeIds.forEach(id => {
                    const el = document.getElementById(id);
                    if (el) {
                        el.innerText = el.dataset.original;
                    }
                });
            } else if (selectedSensor == 'all') {
                Object.entries(panelMapSensorAll).forEach(([label, panelId]) => {
                    const iframeId = `grafanaIframe${label}_sensorAll`;
                    const iframe = document.getElementById(iframeId);
                    if (!iframe) return;

                    const defaultSrc =
                        `http://labai.polinema.ac.id:3010/d-solo/eempvyqjk5csgf/website-visualisasi-data?orgId=1&timezone=browser&theme=light&panelId=${panelId}&__feature=dashboardSceneSolo`;
                    iframe.src = defaultSrc;
                });

                const titleIframeIds = [
                    'titleTemperature_sensorAll',
                    'titleHumidity_sensorAll',
                    'titleConductivity_sensorAll',
                    'titlepH_sensorAll',
                    'titleNitrogen_sensorAll',
                    'titlePhosphorus_sensorAll',
                    'titlePotassium_sensorAll',
                ];

                titleIframeIds.forEach(id => {
                    const el = document.getElementById(id);
                    if (el) {
                        el.innerText = el.dataset.original;
                    }
                });
            }
        }
    </script>
@endpush
