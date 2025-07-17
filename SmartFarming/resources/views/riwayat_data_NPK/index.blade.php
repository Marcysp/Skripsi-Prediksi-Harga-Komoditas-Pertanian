@extends('layout.template')
@section('title', 'Riwayat NPK | Agrilink Vocpro')
@section('content')
    <div class="container-fluid">
        <div class="row">
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
        <div class="card">
            <div class="card-body">
                <div class="responsive-table-wrapper">
                    <table class="table align-middle table-striped table-bordered" id="table_riwayat_data_npk">
                        <thead>
                            <tr class="text-center">
                                <th class="text-center">No</th>
                                <th class="text-center">Suhu</th>
                                <th class="text-center">Kelembapan</th>
                                <th class="text-center">Konduktivitas</th>
                                <th class="text-center">pH</th>
                                <th class="text-center">Nitrogen</th>
                                <th class="text-center">Fosfor</th>
                                <th class="text-center">Kalium</th>
                                <th class="text-center">Sensor</th>
                                <th class="text-center">Tanggal</th>
                            </tr>
                        </thead>
                    </table>
                </div>
            </div>
        </div>
    </div>
@endsection
@push('js')
    <script>
        $(document).ready(function() {
            // Ambil nilai dari URL jika tersedia
            let startDate = getQueryParam('start_date') || '';
            let endDate = getQueryParam('end_date') || '';
            let selectedSensor = getQueryParam('selected_sensor_npk') || '';

            // Set nilai awal dropdown jika ada di URL
            if (selectedSensor) {
                $('#selected_sensor_npk').val(selectedSensor).trigger('change');
            }

            // Tampilkan tanggal di input jika tersedia
            if (startDate && endDate) {
                $('#daterange').val(moment(startDate).format('DD-MM-YY') + ' → ' + moment(endDate).format(
                    'DD-MM-YY'));
            }

            const datadhts = $('#table_riwayat_data_npk').DataTable({
                searching: false,
                processing: true,
                serverSide: true,
                responsive: true,
                ajax: {
                    url: "{{ route('riwayatDataNPK.list') }}",
                    type: "POST",
                    data: function(d) {
                        d.start_date = startDate;
                        d.end_date = endDate;
                        d.selected_sensor_npk = $('#selected_sensor_npk').val();
                    }
                },
                columns: [{
                        data: "DT_RowIndex",
                        className: "text-center",
                        orderable: false,
                        searchable: false
                    },
                    {
                        data: "temperature",
                        className: "text-center",
                        orderable: false,
                        searchable: false
                    },
                    {
                        data: "humidity",
                        className: "text-center",
                        orderable: false,
                        searchable: false
                    },
                    {
                        data: "conductivity",
                        className: "text-center",
                        orderable: false,
                        searchable: false
                    },
                    {
                        data: "ph",
                        className: "text-center",
                        orderable: false,
                        searchable: false
                    },
                    {
                        data: "nitrogen",
                        className: "text-center",
                        orderable: false,
                        searchable: false
                    },
                    {
                        data: "phosphorus",
                        className: "text-center",
                        orderable: false,
                        searchable: false
                    },
                    {
                        data: "potassium",
                        className: "text-center",
                        orderable: false,
                        searchable: false
                    },
                    {
                        data: "sensors.public_name",
                        className: "text-center",
                        orderable: false,
                        searchable: false
                    },
                    {
                        data: "created_at",
                        className: "text-center",
                        orderable: false,
                        searchable: false
                    }
                ]
            });

            // Inisialisasi date range picker
            $('#daterange').daterangepicker({
                opens: 'left',
                autoUpdateInput: false,
                locale: {
                    applyLabel: 'Pilih',
                    cancelLabel: 'Batal',
                    format: 'DD-MM-YYYY',
                    daysOfWeek: ['Min', 'Sen', 'Sel', 'Rab', 'Kam', 'Jum', 'Sab'],
                    monthNames: ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
                        'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'
                    ],
                }
            });

            $('#daterange').on('apply.daterangepicker', function(ev, picker) {
                startDate = picker.startDate.format('YYYY-MM-DD');
                endDate = picker.endDate.format('YYYY-MM-DD');
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

                $(this).val(picker.startDate.format('DD-MM-YY') + ' → ' + picker.endDate.format(
                    'DD-MM-YY'));
                updateURL();
                datadhts.draw();
            });

            $('#daterange').on('cancel.daterangepicker', function(ev, picker) {
                $(this).val('');
                startDate = '';
                endDate = '';
                updateURL();
                datadhts.draw();
            });

            $('#selected_sensor_npk').on('change', function() {
                updateURL();
                datadhts.draw();
            });

            // -----------------------------
            // Helper Functions
            // -----------------------------

            function getQueryParam(param) {
                const urlParams = new URLSearchParams(window.location.search);
                return urlParams.get(param);
            }

            function updateURL() {
                const newUrl = new URL(window.location.href);
                if (startDate && endDate) {
                    newUrl.searchParams.set('start_date', startDate);
                    newUrl.searchParams.set('end_date', endDate);
                } else {
                    newUrl.searchParams.delete('start_date');
                    newUrl.searchParams.delete('end_date');
                }

                const selectedSensor = $('#selected_sensor_npk').val();
                if (selectedSensor) {
                    newUrl.searchParams.set('selected_sensor_npk', selectedSensor);
                } else {
                    newUrl.searchParams.delete('selected_sensor_npk');
                }

                // Update URL tanpa reload
                window.history.replaceState({}, '', newUrl);
            }
        });
    </script>
@endpush
