<?php

namespace App\Http\Controllers;

use Carbon\Carbon;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;

class MonitoringSensorDHTController extends Controller
{
    public function index()
    {
        $breadcrumb = (object) [
            'title' => 'Pemantauan Sensor DHT',
            'paragraph' => 'Pantau data sensor secara real-time untuk menjaga kondisi optimal pada greenhouse.',
            'list' => [
                ['label' => 'Dasbor', 'url' => route('dashboard.index')],
                ['label' => 'Sensor DHT'],
            ]
        ];
        $activeMenu = 'monitoringSensorDHT';

        Carbon::setLocale('id');
        $latestData = DB::table('sensor_readings')
            ->where('sensor_id', 1)
            ->orderByDesc('created_at')
            ->limit(1)
            ->get();
        $dataDHT = collect();
        foreach ($latestData as $data) {
            $payload = json_decode($data->payload, true);
            $createdAt = Carbon::parse($data->created_at)->timezone('Asia/Jakarta');
            $jamMenit = $createdAt->format('H:i');
            $dataDHT->push([
                'label' => 'Update Terakhir',
                'value' => $jamMenit,
                'unit' => '',
                'icon' => 'bi-calendar'
            ]);
            $dataDHT->push([
                'label' => 'Suhu Ruangan',
                'value' => $payload['viciTemperature'] ?? '-',
                'unit' => '°C',
                'icon' => 'bi-thermometer-half'
            ]);
            $dataDHT->push([
                'label' => 'Kelembapan Ruangan',
                'value' => $payload['viciHumidity'] ?? '-',
                'unit' => '%',
                'icon' => 'bi-droplet-half'
            ]);
            $dataDHT->push([
                'label' => 'Intensitas Cahaya',
                'value' => $payload['viciLuminosity'] ?? '-',
                'unit' => 'lux',
                'icon' => 'bi-brightness-high-fill'
            ]);
        }

        return view('monitoring_sensor_DHT.index', compact(
            'breadcrumb',
            'activeMenu',
            'dataDHT',
        ));
    }
}
