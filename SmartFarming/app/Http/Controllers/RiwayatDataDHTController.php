<?php

namespace App\Http\Controllers;

use App\Models\DHTSModel;
use App\Models\SensorReadingsModel;
use Illuminate\Http\Request;
use Yajra\DataTables\Facades\DataTables;

class RiwayatDataDHTController extends Controller
{
    public function index()
    {
        $breadcrumb = (object) [
            'title' => 'Riwayat Data Sensor DHT',
            'paragraph' => 'Pusat informasi kondisi greenhouse Anda secara menyeluruh',
            'list' => [
                ['label' => 'Dasbor', 'url' => route('dashboard.index')],
                ['label' => 'Sensor DHT'],
            ]
        ];
        $activeMenu = 'riwayatDataDHT';
        return view('riwayat_data_DHT.index', compact(
            'breadcrumb',
            'activeMenu',
        ));
    }

    public function list(Request $request)
    {
        $dhts = SensorReadingsModel::select(['id', 'payload', 'sensor_id', 'created_at'])
            ->with(['sensor'])
            ->where('sensor_id', 1)
            ->orderBy('created_at', 'desc');

        // ✅ Tambahkan filter tanggal jika tersedia
        if ($request->start_date && $request->end_date) {
            $dhts->whereBetween('created_at', [
                $request->start_date . ' 00:00:00',
                $request->end_date . ' 23:59:59',
            ]);
        }

        return DataTables::of($dhts)
            ->addIndexColumn()

            // Ambil temperature dari payload
            ->editColumn('temperature', function ($row) {
                return ($row->payload['viciTemperature'] ?? 'null') . ' °C';
            })

            // Ambil humidity dari payload
            ->editColumn('humidity', function ($row) {
                return ($row->payload['viciHumidity'] ?? 'null') . ' %';
            })

            // Ambil luminosity dari payload
            ->editColumn('luminosity', function ($row) {
                return ($row->payload['viciLuminosity'] ?? 'null') . ' lux';
            })

            // Format tanggal
            ->editColumn('created_at', function ($row) {
                return $row->created_at->format('d-m-Y H:i:s');
            })

            // Tampilkan sensor name dari relasi jika ada
            ->addColumn('sensors.public_name', function ($row) {
                return optional($row->sensor)->public_name;
            })

            ->make(true);
    }
}
