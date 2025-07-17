<?php

namespace App\Http\Controllers;

use App\Models\NPKSModel;
use App\Models\SensorReadingsModel;
use App\Models\SensorsModel;
use Illuminate\Http\Request;
use Yajra\DataTables\Facades\DataTables;

class RiwayatDataNPKController extends Controller
{
    public function index(Request $request)
    {
        $breadcrumb = (object) [
            'title' => 'Riwayat Data Sensor NPK',
            'paragraph' => 'Pusat informasi kondisi greenhouse Anda secara menyeluruh',
            'list' => [
                ['label' => 'Dasbor', 'url' => route('dashboard.index')],
                ['label' => 'Sensor NPK'],
            ]
        ];
        $activeMenu = 'riwayatDataNPK';

        $sensor_npk = SensorsModel::whereNotNull('bed_location_id')->get();
        $selectedSensor = $request->input('selected_sensor_npk');
        session(['selected_sensor_npk' => $selectedSensor]);

        return view('riwayat_data_NPK.index', compact(
            'breadcrumb',
            'activeMenu',
            'sensor_npk',
        ));
    }

    public function list(Request $request)
    {
        $npks = SensorReadingsModel::select(['id', 'payload', 'sensor_id', 'created_at'])
            ->with(['sensor'])
            ->whereIn('sensor_id', [2, 3])
            ->orderBy('created_at', 'desc');

        // ✅ Tambahkan filter tanggal jika tersedia
        if ($request->start_date && $request->end_date) {
            $npks->whereBetween('created_at', [
                $request->start_date . ' 00:00:00',
                $request->end_date . ' 23:59:59',
            ]);
        }

        if ($request->has('selected_sensor_npk') && $request->selected_sensor_npk != '') {
            $npks->where('sensor_id', $request->selected_sensor_npk);
        }

        return DataTables::of($npks)
            ->addIndexColumn()
            ->editColumn('temperature', function ($row) {
                return ($row->payload['soilTemperature'] ?? 'null') . ' °C';
            })
            ->editColumn('humidity', function ($row) {
                return ($row->payload['soilHumidity'] ?? 'null') . ' %';
            })
            ->editColumn('conductivity', function ($row) {
                return ($row->payload['soilConductivity'] ?? 'null') . ' μS/cm';
            })
            ->editColumn('ph', function ($row) {
                return ($row->payload['soilPh'] ?? 'null') . ' pH';
            })
            ->editColumn('nitrogen', function ($row) {
                return ($row->payload['soilNitrogen'] ?? 'null') . ' mg/kg';
            })
            ->editColumn('phosphorus', function ($row) {
                return ($row->payload['soilPhosphorus'] ?? 'null') . ' mg/kg';
            })
            ->editColumn('potassium', function ($row) {
                return ($row->payload['soilPotassium'] ?? 'null') . ' mg/kg';
            })
            ->editColumn('created_at', function ($row) {
                return $row->created_at->format('d-m-Y H:i:s');
            })
            ->addColumn('sensors.public_name', function ($row) {
                return optional($row->sensor)->public_name;
            })
            ->make(true);
    }

    public function storeFilter(Request $request)
    {
        $request->session()->put('selected_sensor', $request->input('selected_sensor_npk'));
        return redirect()->route('riwayatDataNPK.index');
    }
}
