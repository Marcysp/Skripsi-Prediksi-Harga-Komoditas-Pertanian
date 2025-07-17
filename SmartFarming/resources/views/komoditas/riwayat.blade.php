@extends('layout.template')
@section('title', 'Monitoring Cuaca | Agrilink Vocpro')
@section('content')
    <div class="container-fluid">
        <div class="d-flex align-items-start form-group row mb-0">
            <div class="form-group col-12 col-lg-3">
                <label for="start_date" class="form-label">Tanggal:</label>
                <input type="text" class="form-control" name="daterange" id="daterange" placeholder="Masukkan tanggal">
            </div>
            <div class="form-group col-12 col-lg-3 mb-0">
                <label for="sensor_npk" class="form-label">Kab/Kota :</label>
                <div class="form-group">
                    <select class="choices form-select" name="selected_kabkota" id="selected_kabkota" required>
                        <option value="">- Kab/Kota -</option>
                        @foreach ($data as $item)
                            <option value="{{ $item->id }}" {{ $selectedKabkota == $item->id ? 'selected' : '' }}>
                                {{ $item->kab_nama }}
                            </option>
                        @endforeach
                    </select>
                </div>
            </div>
            <div class="form-group col-12 col-lg-3 mb-0">
                <label for="sensor_npk" class="form-label">Pasar :</label>
                <div class="form-group">
                    <select class="choices form-select" name="selected_pasar" id="selected_pasar" required>
                        <option value="">- Pilih Pasar -</option>
                        {{-- akan diisi secara dinamis oleh JavaScript --}}
                    </select>
                </div>
            </div>
        </div>
    </div>
    <div class="container-fluid">
        <div class="card">
            <div class="card-body">
                <h5 id="judulHarga">
                    Harga rata-rata di Jawa Timur {{ \Carbon\Carbon::now()->format('Y-m-d') }}
                </h5>

                <div class="responsive-table-wrapper">
                    <table class="table table-hover" id="table_riwayat">
                        <thead>
                            <tr>
                                <th>No</th>
                                <th>Nama Bahan Pokok</th>
                                <th>Satuan</th>
                                <th>Harga Kemarin</th>
                                <th>Harga Sekarang</th>
                                <th>Perubahan (Rp)</th>
                                <th>Perubahan (%)</th>
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
                            className: 'text-center'
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
            const kabkotaSelect = document.getElementById('selected_kabkota');
            const pasarSelect = document.getElementById('selected_pasar');

            kabkotaSelect.addEventListener('change', function() {
                const kabkotaId = this.value;

                // Reset pasar dropdown
                pasarSelect.innerHTML = '<option value="">- Pilih Pasar -</option>';

                if (kabkotaId) {
                    fetch(`/komoditas/get-pasar/${kabkotaId}`)
                        .then(response => response.json())
                        .then(data => {
                            data.forEach(pasar => {
                                const option = document.createElement('option');
                                option.value = pasar.id;
                                option.text = pasar.psr_nama;
                                pasarSelect.appendChild(option);
                            });
                        })
                        .catch(error => {
                            console.error('Gagal memuat data pasar:', error);
                        });
                }
            });
        });
    </script>
    <script>
        function loadRiwayatTable(pasarId = '', tanggal = '') {
            $('#table_riwayat').DataTable({
                destroy: true,
                processing: true,
                paging: false,
                ajax: {
                    url: "{{ route('riwayat.data') }}",
                    data: {
                        pasar_id: pasarId,
                        tanggal: tanggal
                    }
                },
                columns: [{
                        data: 'no',
                        className: 'text-center'
                    },
                    {
                        data: 'komoditas_nama',
                        className: 'text-center'
                    },
                    {
                        data: 'satuan',
                        className: 'text-center'
                    },
                    {
                        data: 'harga_kemarin',
                        className: 'text-center'
                    },
                    {
                        data: 'harga_sekarang',
                        className: 'text-center'
                    },
                    {
                        data: 'perubahan',
                        className: 'text-center'
                    },
                    {
                        data: 'perubahan%',
                        className: 'text-center',
                        render: function(data, type, row) {
                            let value = parseFloat(data);
                            let color = 'black';

                            if (value > 0) {
                                color = 'red';
                            } else if (value < 0) {
                                color = 'green';
                            }

                            return `<span style="color:${color}">${value}%</span>`;
                        }
                    }
                ]
            });
        }

        $(document).ready(function() {
            document.getElementById('selected_kabkota').selectedIndex = 0;
            const dateInput = $('#daterange');
            const pasarInput = $('#selected_pasar');

            function reloadTable() {
                const tanggal = dateInput.val();
                const pasarId = pasarInput.val();
                loadRiwayatTable(pasarId, tanggal);
            }

            dateInput.on('change', reloadTable);
            pasarInput.on('change', reloadTable);

            dateInput.on('change', function() {
                reloadTable();
                updateJudulHarga();
            });
            pasarInput.on('change', function() {
                reloadTable();
                updateJudulHarga();
            });

            // Load awal
            reloadTable();
            updateJudulHarga();
        });

        function updateJudulHarga() {
            const pasarSelect = document.getElementById('selected_pasar');
            const kabkotaSelect = document.getElementById('selected_kabkota');
            const tanggal = document.getElementById('daterange').value;

            const pasarText = pasarSelect.options[pasarSelect.selectedIndex]?.text;
            const kabkotaText = kabkotaSelect.options[kabkotaSelect.selectedIndex]?.text;

            const h5 = document.getElementById('judulHarga');

            if (pasarSelect.value) {
                h5.textContent = `Harga ${pasarText} ${kabkotaText} ${tanggal}`;
            } else {
                h5.textContent = `Harga rata-rata di Jawa Timur ${tanggal}`;
            }
        }
    </script>
@endpush
