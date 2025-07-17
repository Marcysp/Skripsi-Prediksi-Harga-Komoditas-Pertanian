@extends('layout.template')
@section('title', 'Riwayat DHT | Agrilink Vocpro')
@section('content')
    <div class="container-fluid">
        <div class="form-group col-12 col-lg-3 ms-auto">
            <label for="start_date" class="form-label">Filter Tanggal:</label>
            <input type="text" class="form-control" name="daterange" id="daterange" placeholder="Masukkan tanggal">
        </div>
        <div class="card">
            <div class="card-body">
                <div class="responsive-table-wrapper">
                    <table class="table align-middle table-striped table-bordered" id="table_riwayat_data_cuaca">
                        <thead>
                            <tr class="text-center">
                                <th class="text-center">No</th>
                                <th class="text-center">Tanggal</th>
                                <th class="text-center">Suhu</th>
                                <th class="text-center">Tutupan Awan</th>
                                <th class="text-center">Kecepatan Angin</th>
                                <th class="text-center">Cuaca</th>
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
            let startDate = getQueryParam('start_date') || '';
            let endDate = getQueryParam('end_date') || '';

            // Inisialisasi DataTable
            const datadhts = $('#table_riwayat_data_cuaca').DataTable({
                searching: false,
                processing: true,
                serverSide: true,
                responsive: true,
                ajax: {
                    url: "{{ route('riwayatDataCuaca.list') }}",
                    type: "POST",
                    data: function(d) {
                        d.start_date = startDate;
                        d.end_date = endDate;
                    }
                },
                columns: [{
                        data: "DT_RowIndex",
                        className: "text-center",
                        orderable: false,
                        searchable: false
                    },
                    {
                        data: "time",
                        className: "text-center",
                        orderable: false,
                        searchable: false
                    },
                    {
                        data: "temperature_2m",
                        className: "text-center",
                        orderable: false,
                        searchable: false
                    },
                    {
                        data: "cloud_cover",
                        className: "text-center",
                        orderable: false,
                        searchable: false
                    },
                    {
                        data: "wind_speed_10m",
                        className: "text-center",
                        orderable: false,
                        searchable: false
                    },
                    {
                        data: "weather_code",
                        className: "text-center",
                        orderable: false,
                        searchable: false
                    }
                ]
            });

            // Tampilkan ulang tanggal terpilih jika tersedia di URL
            if (startDate && endDate) {
                $('#daterange').val(moment(startDate).format('DD-MM-YY') + ' → ' + moment(endDate).format(
                    'DD-MM-YY'));
            }

            // Inisialisasi Daterangepicker
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

            // Saat tombol "Pilih" ditekan
            $('#daterange').on('apply.daterangepicker', function(ev, picker) {
                startDate = picker.startDate.format('YYYY-MM-DD');
                endDate = picker.endDate.format('YYYY-MM-DD');

                $(this).val(picker.startDate.format('DD-MM-YY') + ' → ' + picker.endDate.format(
                    'DD-MM-YY'));
                const newUrl = updateQueryString(window.location.href, startDate, endDate);
                window.history.replaceState(null, '', newUrl);

                datadhts.draw();
            });

            // Saat "Batal" ditekan
            $('#daterange').on('cancel.daterangepicker', function(ev, picker) {
                $(this).val('');
                startDate = '';
                endDate = '';
                const newUrl = removeQueryString(window.location.href);
                window.history.replaceState(null, '', newUrl);
                datadhts.draw();
            });

            // Helper: Ambil query param dari URL
            function getQueryParam(param) {
                const urlParams = new URLSearchParams(window.location.search);
                return urlParams.get(param);
            }

            // Helper: Update query string dengan tanggal baru
            function updateQueryString(url, start, end) {
                const urlObj = new URL(url);
                urlObj.searchParams.set('start_date', start);
                urlObj.searchParams.set('end_date', end);
                return urlObj.toString();
            }

            // Helper: Hapus query string
            function removeQueryString(url) {
                const urlObj = new URL(url);
                urlObj.searchParams.delete('start_date');
                urlObj.searchParams.delete('end_date');
                return urlObj.pathname;
            }
        });
    </script>
@endpush
