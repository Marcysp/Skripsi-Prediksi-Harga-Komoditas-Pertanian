@extends('layout.template')
@section('title', 'Monitoring NPK | Agrilink Vocpro')
@section('content')
    <div class="container-fluid">
        <div class="d-flex align-items-end form-group row mb-0">
            <div class="form-group col-12 col-lg-3 mb-0">
                <label for="sensor_npk" class="form-label">Komoditas :</label>
                <div class="form-group">
                    <select class="choices form-select" name="selected_komoditas" id="selected_komoditas" required>
                        <option value="">- Komoditas -</option>
                        @foreach ($data as $item)
                            <option value="{{ $item }}" {{ $selectedKomoditas == $item ? 'selected' : '' }}>
                                {{ $item }}
                            </option>
                        @endforeach
                    </select>
                </div>
            </div>
            <div class="form-group col-12 col-lg-3 ms-auto">
                <label for="start_date" class="form-label">Tanggal diprediksi:</label>
                <input type="text" class="form-control" name="daterange" id="daterange" placeholder="Masukkan tanggal">
            </div>
        </div>
        <div class="card" style="border-color: #CED4DA">
            <div class="card-body">
                <div class="ratio ratio-16x9">
                    <iframe id="grafanaFrame"
                        src="http://localhost:3010/d-solo/aekxvuuuvs1z4d/grafik-prediksi?orgId=1&from=1747146651126&to=1763044251126&timezone=browser&var-nama_komoditas=Cabe%20Merah%20Besar&var-limit=60&var-tanggal=2025-07-13&refresh=1d&showCategory=Legend&panelId=3&__feature.dashboardSceneSolo"
                        width="450" height="200" frameborder="0"></iframe>
                </div>
            </div>
        </div>
    </div>
    <div class="container-fluid">
        <div class="card">
            <div class="card-body">
                <div class="responsive-table-wrapper">
                    <table class="table align-middle table-striped table-bordered" id="tabel_prediksi">
                        <thead>
                            <tr class="text-center">
                                <th class="text-center">No</th>
                                <th class="text-center">Tanggal</th>
                                <th class="text-center">Prediksi</th>
                            </tr>
                        </thead>
                    </table>
                </div>
            </div>
        </div>
    </div>
@endsection
@push('css')
@endpush
@push('js')
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const select = document.getElementById('selected_komoditas');
            const iframe = document.getElementById('grafanaFrame');
            let selectedDate = moment().format('YYYY-MM-DD'); // default hari ini
            let table; // untuk referensi DataTable

            function getSelectedKomoditas() {
                return select.value || 'Tomat Merah';
            }

            function updateIframe() {
                const tanggal = selectedDate;
                const selectedKomoditas = getSelectedKomoditas();

                const baseGrafanaURL = "http://labai.polinema.ac.id:3010/d-solo/aekxvuuuvs1z4d/tes-grafik";

                const now = new Date();
                const fromDate = new Date(now);
                fromDate.setMonth(fromDate.getMonth() - 2);
                const toDate = new Date(now);
                toDate.setMonth(toDate.getMonth() + 4);

                const from = fromDate.getTime();
                const to = toDate.getTime();

                const params = new URLSearchParams({
                    orgId: 1,
                    from: from,
                    to: to,
                    timezone: "browser",
                    'var-nama_komoditas': selectedKomoditas,
                    'var-limit': 60,
                    'var-tanggal': tanggal,
                    theme: "light",
                    panelId: 3,
                    __feature: "dashboardSceneSolo"
                });

                iframe.src = `${baseGrafanaURL}?${params.toString()}`;
            }

            function initTable() {
                table = $('#tabel_prediksi').DataTable({
                    processing: true,
                    serverSide: false,
                    ajax: {
                        url: "{{ route('prediksi.data') }}",
                        data: function(d) {
                            d.tanggal = selectedDate;
                            d.komoditas = getSelectedKomoditas();
                        }
                    },
                    columns: [{
                            data: 'no',
                            className: 'text-center'
                        },
                        {
                            data: 'tanggal',
                            className: 'text-center'
                        },
                        {
                            data: 'prediksi',
                            className: 'text-center',
                            render: function(data, type, row) {
                                if (data == null) return '-';

                                const rounded = Math.floor(data / 1000) * 1000;
                                const decimal = (data - rounded).toFixed(3).split('.')[
                                1]; // ambil 3 digit pecahan

                                return 'Rp ' + rounded.toLocaleString('id-ID') + ',' + decimal;
                            }
                        }
                    ]
                });
            }

            function reloadTable() {
                table.ajax.reload();
            }

            $('#daterange').daterangepicker({
                singleDatePicker: true,
                showDropdowns: true,
                autoUpdateInput: true,
                locale: {
                    format: 'YYYY-MM-DD',
                    applyLabel: 'Pilih',
                    cancelLabel: 'Batal'
                }
            });

            $('#daterange').on('apply.daterangepicker', function(ev, picker) {
                selectedDate = picker.startDate.format('YYYY-MM-DD');
                updateIframe();
                reloadTable();
            });

            select.addEventListener('change', function() {
                updateIframe();
                reloadTable();
            });

            // Set nilai awal datepicker dan iframe
            $('#daterange').val(selectedDate);
            updateIframe();
            initTable();
        });
    </script>


    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const select = document.getElementById('selected_komoditas');
            const iframe = document.getElementById('grafanaFrame');

            window.updateIframe = function() {
                const selectedKomoditas = select.value || 'Tomat Merah';

                // Gunakan tanggal dari picker
                const tanggal = selectedDate;

                const baseGrafanaURL = "http://labai.polinema.ac.id:3010/d-solo/aekxvuuuvs1z4d/tes-grafik";

                const now = new Date();
                const fromDate = new Date(now);
                fromDate.setMonth(fromDate.getMonth() - 2);
                const toDate = new Date(now);
                toDate.setMonth(toDate.getMonth() + 4);

                const from = fromDate.getTime();
                const to = toDate.getTime();

                const params = new URLSearchParams({
                    orgId: 1,
                    from: from,
                    to: to,
                    timezone: "browser",
                    'var-nama_komoditas': selectedKomoditas,
                    'var-limit': 60,
                    'var-tanggal': tanggal,
                    theme: "light",
                    panelId: 3,
                    __feature: "dashboardSceneSolo"
                });

                iframe.src = `${baseGrafanaURL}?${params.toString()}`;
                console.log(`Updated iframe src: ${iframe.src}`);
            }

            // Update saat pertama kali
            updateIframe();

            // Update saat select berubah
            select.addEventListener('change', updateIframe);
        });
    </script>
@endpush
