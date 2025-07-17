<?php

namespace App\Http\Controllers;

use Carbon\Carbon;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;

class MonitoringCuacaController extends Controller
{
    public function index()
    {
        $breadcrumb = (object) [
            'title' => 'Prediksi Cuaca',
            'paragraph' => 'Prediksi cuaca 7 hari ke depan untuk membantu petani merencanakan tanam, penyiraman, dan pemupukan secara efisien.',
            'list' => [
                ['label' => 'Dasbor', 'url' => route('dashboard.index')],
                ['label' => 'Prediksi Cuaca'],
            ]
        ];
        $activeMenu = 'monitoringCuaca';

        $nowJakarta = Carbon::now('Asia/Jakarta');
        $nowUtc = $nowJakarta->copy()->setTimezone('UTC');
        $latestData = DB::table('weather_data')
            ->whereDate('time', $nowUtc->toDateString()) // tanggal dalam UTC
            ->whereTime('time', '<=', $nowUtc->format('H:i:s')) // waktu dalam UTC
            ->orderByDesc('time')
            ->first();

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

        $weatherIcons = [
            // Cerah
            0 => 'bi-sun-fill',
            1 => 'bi-sun-fill',
            2 => 'bi-cloud-sun-fill',
            3 => 'bi-cloud-fill',

            // Kabut
            45 => 'bi-cloud-fog2-fill',
            48 => 'bi-cloud-fog2-fill',

            // Gerimis
            51 => 'bi-cloud-drizzle-fill',
            53 => 'bi-cloud-drizzle-fill',
            55 => 'bi-cloud-drizzle-fill',
            56 => 'bi-cloud-drizzle-fill',
            57 => 'bi-cloud-drizzle-fill',

            // Hujan
            61 => 'bi-cloud-rain-fill',
            63 => 'bi-cloud-rain-fill',
            65 => 'bi-cloud-rain-fill',
            66 => 'bi-cloud-rain-fill',
            67 => 'bi-cloud-rain-fill',
            80 => 'bi-cloud-rain-fill',
            81 => 'bi-cloud-rain-fill',
            82 => 'bi-cloud-rain-heavy-fill',

            // Salju
            71 => 'bi-cloud-snow-fill',
            73 => 'bi-cloud-snow-fill',
            75 => 'bi-cloud-snow-fill',
            77 => 'bi-cloud-snow-fill',
            85 => 'bi-cloud-snow-fill',
            86 => 'bi-cloud-snow-fill',

            // Petir
            95 => 'bi-cloud-lightning-fill',
            96 => 'bi-cloud-lightning-rain-fill',
            99 => 'bi-cloud-lightning-rain-fill',
        ];

        $weatherData = collect();

        if ($latestData) {
            $deskripsiCuaca = $weatherDescriptions[$latestData->weather_code] ?? 'Tidak diketahui';
            $ikonCuaca = $weatherIcons[$latestData->weather_code] ?? 'bi-question-circle-fill';

            if ($latestData->temperature_2m <= 20) {
                $ikonSuhu = 'bi-thermometer-snow';
            } elseif ($latestData->temperature_2m <= 30) {
                $ikonSuhu = 'bi-thermometer-half';
            } else {
                $ikonSuhu = 'bi-thermometer-sun';
            }

            $weatherData->push([
                'label' => 'Suhu',
                'value' => $latestData->temperature_2m,
                'unit' => '°C',
                'icon' => 'bi ' . $ikonSuhu
            ]);
            $weatherData->push([
                'label' => 'Tutupan Awan',
                'value' => $latestData->cloud_cover,
                'unit' => '%',
                'icon' => 'bi bi-clouds-fill'
            ]);
            $weatherData->push([
                'label' => 'Kecepatan Angin',
                'value' => $latestData->wind_speed_10m,
                'unit' => 'km/h',
                'icon' => 'bi bi-wind'
            ]);
            $weatherData->push([
                'label' => 'Cuaca',
                'value' => $deskripsiCuaca,
                'unit' => '',
                'icon' => 'bi ' . $ikonCuaca
            ]);
        }

        return view('monitoring_cuaca.index', compact(
            'breadcrumb',
            'activeMenu',
            'weatherData',
        ));
    }
}
