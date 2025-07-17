@extends('layout.template')
@section('title', 'Monitoring Cuaca | Agrilink Vocpro')
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
        </div>
        <div class="row">
            <div class="col-12 col-lg-6">
                <div class="card" style="border-color: #CED4DA">
                    <div class="card-body">
                        <h6 id="titleTemperature" data-original="Suhu">Harga rata-rata di Jawa Timur dalam 30 Hari Terakhir
                        </h6>
                        <div class="ratio ratio-16x9">
                            <iframe class="grafana-frame" id="iframe1"
                                src="http://labai.polinema.ac.id:3010/d-solo/eersin3z4iscgd/tren-harga?orgId=1&from=1749772800000&to=1752278400000&timezone=browser&var-komoditas=Cabe%20Merah%20Besar&theme=light&panelId=8&__feature.dashboardSceneSolo"
                                width="450" height="200" frameborder="0"></iframe>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-12 col-lg-6">
                <div class="card" style="border-color: #CED4DA">
                    <div class="card-body">
                        <h6 id="titleCloudCover" data-original="Tutupan Awan">Harga rata-rata bulanan dalam 12 bulan
                            terakhir</h6>
                        <div class="ratio ratio-16x9">
                            <iframe class="grafana-frame" id="iframe2"
                                src="http://labai.polinema.ac.id:3010/d-solo/eersin3z4iscgd/tren-harga?orgId=1&from=1718236800000&to=1752278400000&timezone=browser&var-komoditas=Cabe%20Merah%20Besar&theme=light&panelId=9&__feature.dashboardSceneSolo"
                                width="450" height="200" frameborder="0"></iframe>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="row">
            <div class="col-12 col-lg-6">
                <div class="card" style="border-color: #CED4DA">
                    <div class="card-body">
                        <h6 id="titleTemperature" data-original="Suhu">Rata-rata harga tahunan di Jawa Timur</h6>
                        <div class="ratio ratio-16x9">
                            <iframe class="grafana-frame" id="iframe3"
                                src="http://labai.polinema.ac.id:3010/d-solo/eersin3z4iscgd/tren-harga?orgId=1&from=1609459200000&to=1735689600000&timezone=browser&var-komoditas=Cabe%20Merah%20Besar&theme=light&panelId=7&__feature.dashboardSceneSolo"
                                width="450" height="200" frameborder="0"></iframe>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-12 col-lg-6">
                <div class="card" style="border-color: #CED4DA">
                    <div class="card-body">
                        <h6 id="titleCloudCover" data-original="Tutupan Awan">
                            Harga rata-rata di Jawa Timur pada setiap bulan {{ now()->translatedFormat('F') }}
                        </h6>
                        <div class="ratio ratio-16x9">
                            <iframe class="grafana-frame" id="iframe4"
                                src="http://labai.polinema.ac.id:3010/d-solo/eersin3z4iscgd/tren-harga?orgId=1&from=1609459200000&to=1735689600000&timezone=browser&var-komoditas=Cabe%20Merah%20Besar&showCategory=Panel%20options&theme=light&panelId=5&__feature.dashboardSceneSolo"
                                width="450" height="200" frameborder="0"></iframe>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="card" style="border-color: #CED4DA">
            <div class="card-body">
                <h6 id="titleTemperature" data-original="Suhu">Harga tiap Kabupaten/Kota di Jawa Timur</h6>
                <div class="ratio ratio-16x9">
                    <iframe class="grafana-frame" id="iframe5"
                        src="http://labai.polinema.ac.id:3010/d-solo/eersin3z4iscgd/tren-harga?orgId=1&from=1609459200000&to=1735689600000&timezone=browser&var-komoditas=Cabe%20Merah%20Besar&theme=light&panelId=4&__feature.dashboardSceneSolo"
                        width="450" height="200" frameborder="0"></iframe>
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

            function updateAllIframes() {
                const komoditas = select.value || 'Tomat Merah';

                // Waktu sekarang
                const now = new Date();

                // iframe1: 30 hari terakhir
                const from1 = new Date(now);
                from1.setDate(from1.getDate() - 30);
                const to1 = now;

                // iframe2: 12 bulan terakhir
                const from2 = new Date(now);
                from2.setMonth(from2.getMonth() - 12);
                const to2 = now;

                // iframe3: 7 tahun terakhir (dari awal tahun ini - 7 tahun)
                const from3 = new Date(now.getFullYear() - 7, 0, 1); // Jan 1, 7 tahun lalu
                const to3 = new Date(now.getFullYear(), 11, 31); // 31 Des tahun ini

                // Update iframe1
                const iframe1 = document.getElementById('iframe1');
                const url1 = new URL(iframe1.src);
                url1.searchParams.set('var-komoditas', komoditas);
                url1.searchParams.set('from', from1.getTime());
                url1.searchParams.set('to', to1.getTime());
                iframe1.src = url1.toString();

                // Update iframe2
                const iframe2 = document.getElementById('iframe2');
                const url2 = new URL(iframe2.src);
                url2.searchParams.set('var-komoditas', komoditas);
                url2.searchParams.set('from', from2.getTime());
                url2.searchParams.set('to', to2.getTime());
                iframe2.src = url2.toString();

                // Update iframe3
                const iframe3 = document.getElementById('iframe3');
                const url3 = new URL(iframe3.src);
                url3.searchParams.set('var-komoditas', komoditas);
                url3.searchParams.set('from', from3.getTime());
                url3.searchParams.set('to', to3.getTime());
                iframe3.src = url3.toString();

                // iframe4 dan iframe5 tetap, hanya ganti komoditas
                const iframe4 = document.getElementById('iframe4');
                const url4 = new URL(iframe4.src);
                url4.searchParams.set('var-komoditas', komoditas);
                iframe4.src = url4.toString();

                const iframe5 = document.getElementById('iframe5');
                const url5 = new URL(iframe5.src);
                url5.searchParams.set('var-komoditas', komoditas);
                iframe5.src = url5.toString();
            }

            // Jalankan saat halaman pertama kali dimuat
            updateAllIframes();

            // Update semua iframe saat dropdown komoditas berubah
            select.addEventListener('change', updateAllIframes);
        });
    </script>
@endpush
