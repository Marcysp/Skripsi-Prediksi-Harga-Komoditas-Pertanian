<?php

namespace App\Http\Controllers;

use Carbon\Carbon;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\DB;

class KomoditasController extends Controller
{
    public function prediksi()
    {
        $breadcrumb = (object) [
            'title' => 'Prediksi Harga Komoditas',
            'paragraph' => 'Prediksi harga komoditas untuk membantu petani merencanakan penanaman dan penjualan hasil pertanian.',
            'list' => [
                ['label' => 'Dasbor', 'url' => route('dashboard.index')],
                ['label' => 'Prediksi Harga Komoditas'],
            ]
        ];
        $activeMenu = 'prediksiHarga';
        $data = Cache::remember('komoditas_nama_prediksi', 60, function () {
            return DB::connection('pgsql_secondary')
                ->table('hasil_prediksi')
                ->select('nama_komoditas')
                ->distinct()
                ->get()
                ->pluck('nama_komoditas')
                ->toArray();
        });

        // $data = $this->forecastService->get_nama_komoditas_forecast();
        $data = DB::connection('pgsql_secondary')
            ->table('hasil_prediksi')
            ->select('nama_komoditas')
            ->distinct()
            ->get()
            ->pluck('nama_komoditas')
            ->toArray();
        $selectedKomoditas = request()->input('selected_komoditas', 'Tomat Merah');


        if (!$data) {
            return response()->json(['message' => 'Gagal mengambil data'], 500);
        }
        // Logic for displaying the commodity price prediction
        return view('komoditas.prediksi', compact(
            'breadcrumb',
            'activeMenu',
            'data',
            'selectedKomoditas'
        ));
    }
    public function getPrediksiData(Request $request)
    {
        $tanggal = $request->input('tanggal'); // contoh: '2025-07-10'
        $komoditas = $request->input('komoditas'); // contoh: 'Bawang Merah'

        // Ambil satu baris prediksi dari tanggal dan komoditas tertentu
        $record = DB::connection('pgsql_secondary')
            ->table('hasil_prediksi')
            ->where('tanggal', $tanggal)
            ->where('nama_komoditas', $komoditas)
            ->first();

        if (!$record) {
            return response()->json([
                'data' => []
            ]);
        }

        $rows = [];
        for ($i = 1; $i <= 90; $i++) {
            $val = $record->{"val_$i"};
            $date = \Carbon\Carbon::parse($record->tanggal)->addDays($i - 1)->format('Y-m-d');

            $rows[] = [
                'no' => $i,
                'tanggal' => $date,
                'prediksi' => $val,
            ];
        }

        return response()->json([
            'data' => $rows
        ]);
    }

    public function trend()
    {
        $breadcrumb = (object) [
            'title' => 'Trend Harga Komoditas',
            'paragraph' => 'Lihat trend harga komoditas untuk membantu petani memahami fluktuasi harga dan merencanakan strategi penjualan.',
            'list' => [
                ['label' => 'Dasbor', 'url' => route('dashboard.index')],
                ['label' => 'Trend Harga Komoditas'],
            ]
        ];
        $activeMenu = 'trendHarga';

        $data = Cache::remember('komoditas_nama', 60, function () {
            return DB::connection('pgsql_secondary')
                ->table('komoditas_rata-rata')
                ->select('komoditas_nama')
                ->distinct()
                ->orderBy('komoditas_nama', 'asc')
                ->get()
                ->pluck('komoditas_nama')
                ->toArray();
        });
        $selectedKomoditas = request()->input('selected_sensor_npk', 'Beras Premium');
        // Logic for displaying the commodity price trend
        return view('komoditas.trend', compact(
            'breadcrumb',
            'activeMenu',
            'data',
            'selectedKomoditas'
        ));
    }

    public function riwayat()
    {
        $breadcrumb = (object) [
            'title' => 'Riwayat Harga Komoditas',
            'paragraph' => 'Lihat riwayat harga aktual komoditas.',
            'list' => [
                ['label' => 'Dasbor', 'url' => route('dashboard.index')],
                ['label' => 'Riwayat Harga Komoditas'],
            ]
        ];
        $activeMenu = 'riwayatHarga';

        // Ambil nilai dari query string, misalnya ?selected_kabkota=10
        $selectedKabkota = request()->input('selected_kabkota', '');
        $selectedPasar = request()->input('selected_pasar', '');
        $selectedKomoditas = request()->input('selected_komoditas', '');

        $data = Cache::remember('kab_kota_aktif', 60, function () {
            return DB::connection('pgsql_secondary')
                ->table('kab_kota')
                ->select('id', 'kab_nama')
                ->where('kab_status', true)
                ->orderBy('kab_nama', 'asc')
                ->get();
        });

        return view('komoditas.riwayat', compact(
            'activeMenu',
            'breadcrumb',
            'data',
            'selectedKabkota',
            'selectedPasar',
            'selectedKomoditas'
        ));
    }

    public function getPasarByKabkota($kabkota_id)
    {
        $pasar = DB::connection('pgsql_secondary')
            ->table('pasar')
            ->select('id', 'psr_nama')
            ->where('psr_status', true)
            ->where('kabkota_id', $kabkota_id)
            ->orderBy('psr_nama')
            ->get();
        return response()->json($pasar);
    }
    public function getRiwayatData(Request $request)
    {
        // dd($request->all());
        $pasarId = $request->input('pasar_id');
        $tanggal = $request->input('tanggal') ?? Carbon::now()->format('Y-m-d');
        $tanggalObj = Carbon::parse($tanggal);
        $sebelumnya = $tanggalObj->copy()->subDay()->format('Y-m-d');
        // dd($tanggal, $sebelumnya, $pasarId, $kabkotaId);

        $tanggalList = [$sebelumnya, $tanggal];

        if ($pasarId) {
            $data = DB::connection('pgsql_secondary')
                ->table('komoditas')
                ->where('pasar_id', $pasarId)
                ->whereIn('tanggal', $tanggalList)
                ->select('tanggal', 'komoditas_nama', 'harga', 'satuan')
                ->orderBy('tanggal', 'desc')
                ->orderBy('komoditas_nama', 'asc')
                ->get();
        } else {
            $data = DB::connection('pgsql_secondary')
                ->table('komoditas_rata-rata')
                ->whereIn('tanggal', $tanggalList)
                ->select('tanggal', 'komoditas_nama', 'harga', 'satuan',)
                ->orderBy('tanggal', 'desc')
                ->orderBy('komoditas_nama', 'asc')
                ->get();
        }
        // dd($data);

        $formatted = [];
        foreach ($data as $index => $item) {
            $formatted[] = [
                'no' => $index + 1,
                'komoditas_nama' => $item->komoditas_nama,
                'satuan' => $item->satuan,
                'harga_sekarang' => $item->harga ?? $item->harga,
                'harga_kemarin' => $data->where('tanggal', $sebelumnya)
                    ->where('komoditas_nama', $item->komoditas_nama)
                    ->first()->harga ?? null,
                'perubahan' => $item->harga - ($data->where('tanggal', $sebelumnya)
                    ->where('komoditas_nama', $item->komoditas_nama)
                    ->first()->harga ?? 0),
                'perubahan%' => $item->harga > 0 ? round((($item->harga - ($data->where('tanggal', $sebelumnya)
                    ->where('komoditas_nama', $item->komoditas_nama)
                    ->first()->harga ?? 0)) / $item->harga) * 100, 2) . '%' : '0%',
            ];
        }
        // dd($formatted);

        return response()->json(['data' => $formatted]);
    }
}
