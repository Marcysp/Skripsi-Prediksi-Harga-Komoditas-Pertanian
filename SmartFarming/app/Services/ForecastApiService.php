<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;

class ForecastApiService
{
    protected $baseUrl;

    public function __construct()
    {
        $this->baseUrl = 'http://127.0.0.1:5000/api/forecast/';
    }

    /**
     * Ambil data forecast berdasarkan nama komoditas.
     *
     * @return array|null
     */
    public function get_nama_komoditas_forecast(): ?array
    {
        $url = $this->baseUrl . 'nama_komoditas';

        $response = Http::get($url);

        if ($response->successful()) {
            return $response->json();
        }

        // Jika error, bisa log atau lempar exception
        return null;
    }
}
