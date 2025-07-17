<?php

namespace App\Http\Controllers;

use App\Models\CuacaModel;
use Carbon\Carbon;
use Illuminate\Http\Request;
use Yajra\DataTables\Facades\DataTables;

class RiwayatDataCuacaController extends Controller
{
    public function index()
    {
        $breadcrumb = (object) [
            'title' => 'Riwayat Data Cuaca',
            'paragraph' => 'Menampilkan riwayat data cuaca yang direkam setiap 7 hari. Gunakan filter tanggal untuk melihat data sesuai periode yang diinginkan.',
            'list' => [
                ['label' => 'Dasbor', 'url' => route('dashboard.index')],
                ['label' => 'Riwayat Cuaca'],
            ]
        ];
        $activeMenu = 'riwayatDataCuaca';
        return view('riwayat_data_cuaca.index', compact(
            'breadcrumb',
            'activeMenu',
        ));
    }

    public function list(Request $request)
    {
        $cuaca = CuacaModel::select(['id', 'time', 'temperature_2m', 'weather_code', 'cloud_cover', 'wind_speed_10m'])
            ->orderBy('time', 'desc');

        // ✅ Tambahkan filter tanggal jika tersedia
        if ($request->start_date && $request->end_date) {
            $cuaca->whereBetween('time', [
                $request->start_date . ' 00:00:00',
                $request->end_date . ' 23:59:59',
            ]);
        }

        $weatherDescriptions = [
            0 => 'Cerah',
            1 => 'Cerah berawan',
            2 => 'Berawan',
            3 => 'Mendung',
            45 => 'Kabut',
            48 => 'Kabut dingin',
            51 => 'Gerimis',
            53 => 'Gerimis sedang',
            55 => 'Gerimis lebat',
            56 => 'Gerimis beku',
            57 => 'Gerimis beku lebat',
            61 => 'Hujan ringan',
            63 => 'Hujan',
            65 => 'Hujan lebat',
            66 => 'Hujan es ringan',
            67 => 'Hujan es',
            71 => 'Salju ringan',
            73 => 'Salju',
            75 => 'Salju lebat',
            77 => 'Butiran salju',
            80 => 'Hujan sekejap',
            81 => 'Hujan sebentar',
            82 => 'Hujan deras',
            85 => 'Salju sebentar',
            86 => 'Salju deras',
            95 => 'Petir',
            96 => 'Petir + es ringan',
            99 => 'Petir + es lebat',
        ];

        return DataTables::of($cuaca)
            ->addIndexColumn()

            // Ambil temperature dari payload
            ->editColumn('temperature_2m', function ($row) {
                return ($row->temperature_2m ?? 'null') . ' °C';
            })

            // Ambil humidity dari payload
            ->editColumn('cloud_cover', function ($row) {
                return ($row->cloud_cover ?? 'null') . ' %';
            })

            // Ambil luminosity dari payload
            ->editColumn('wind_speed_10m', function ($row) {
                return ($row->wind_speed_10m ?? 'null') . ' km/h';
            })

            // Format tanggal
            ->editColumn('time', function ($row) {
                return Carbon::parse($row->time)->timezone('Asia/Jakarta')->format('d-m-Y H:i:s');
            })

            ->editColumn('weather_code', function ($row) use ($weatherDescriptions) {
                return $weatherDescriptions[$row->weather_code] ?? 'Tidak diketahui';
            })

            ->make(true);
    }
}
