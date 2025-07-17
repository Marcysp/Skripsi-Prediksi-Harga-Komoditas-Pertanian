<?php

use App\Http\Controllers\CuacaTerkiniController;
use App\Http\Controllers\DashboardController;
use App\Http\Controllers\KomoditasController;
use App\Http\Controllers\LoginController;
use App\Http\Controllers\MonitoringCuacaController;
use App\Http\Controllers\MonitoringSensorDHTController;
use App\Http\Controllers\MonitoringSensorNPKController;
use App\Http\Controllers\ProfilController;
use App\Http\Controllers\RiwayatDataCuacaController;
use App\Http\Controllers\RiwayatDataDHTController;
use App\Http\Controllers\RiwayatDataNPKController;
use App\Http\Controllers\UserController;
use Illuminate\Support\Facades\Route;

/*
|--------------------------------------------------------------------------
| Web Routes
|--------------------------------------------------------------------------
|
| Here is where you can register web routes for your application. These
| routes are loaded by the RouteServiceProvider and all of them will
| be assigned to the "web" middleware group. Make something great!
|
*/

Route::middleware(['guest.custom'])->group(function () {
    Route::get('/', [LoginController::class, 'index'])->name('login.index');
    Route::post('/login', [LoginController::class, 'authenticate'])->name('login.authenticate');
});

Route::middleware(['checkrole:ADMN,USER', 'prevent_back'])->group(function () {
    Route::post('/logout', [LoginController::class, 'logout'])->name('logout');
    Route::get('/dashboard', [DashboardController::class, 'index'])->name('dashboard.index');
    Route::group(['prefix' => 'monitoringSensor'], function () {
        Route::get('/NPK', [MonitoringSensorNPKController::class, 'index'])->name('monitoringSensorNPK.index');
        Route::get('/DHT', [MonitoringSensorDHTController::class, 'index'])->name('monitoringSensorDHT.index');
    });
    Route::get('/monitoringCuaca', [MonitoringCuacaController::class, 'index'])->name('monitoringCuaca.index');
    Route::get('/monitoringCuacaRealtime', [CuacaTerkiniController::class, 'index'])->name('cuacaTerkini.index');
    Route::group(['prefix' => 'riwayatDataDHT'], function () {
        Route::get('/', [RiwayatDataDHTController::class, 'index'])->name('riwayatDataDHT.index');
        Route::post('/list', [RiwayatDataDHTController::class, 'list'])->name('riwayatDataDHT.list');
    });
    Route::group(['prefix' => 'riwayatDataNPK'], function () {
        Route::get('/', [RiwayatDataNPKController::class, 'index'])->name('riwayatDataNPK.index');
        Route::post('/list', [RiwayatDataNPKController::class, 'list'])->name('riwayatDataNPK.list');
    });
    Route::group(['prefix' => 'riwayatDataCuaca'], function () {
        Route::get('/', [RiwayatDataCuacaController::class, 'index'])->name('riwayatDataCuaca.index');
        Route::post('/list', [RiwayatDataCuacaController::class, 'list'])->name('riwayatDataCuaca.list');
    });
    Route::group(['prefix' => 'profil'], function () {
        Route::get('/', [ProfilController::class, 'index'])->name('profil.index');
        Route::get('/edit', [ProfilController::class, 'edit'])->name('profil.edit');
        Route::get('/update', [ProfilController::class, 'update'])->name('profil.update');
    });
    Route::group(['prefix' => 'komoditas'], function () {
        Route::get('/prediksi', [KomoditasController::class, 'prediksi'])->name('komoditas.prediksi');
        Route::get('/trend', [KomoditasController::class, 'trend'])->name('komoditas.trend');
        Route::get('/riwayat', [KomoditasController::class, 'riwayat'])->name('komoditas.riwayat');
        Route::get('/prediksi/data', [KomoditasController::class, 'getPrediksiData'])->name('prediksi.data');
        Route::get('/get-pasar/{kabkota_id}', [KomoditasController::class, 'getPasarByKabkota'])->name('get.pasar.by.kabkota');
        Route::get('/riwayat-data', [KomoditasController::class, 'getRiwayatData'])->name('riwayat.data');

    });
}); // untuk user dan admin
Route::middleware(['checkrole:ADMN', 'prevent_back'])->group(function () {
    Route::group(['prefix' => 'kelolaPengguna'], function () {
        Route::get('/', [UserController::class, 'index'])->name('kelolaPengguna.index');
        Route::post('/list', [UserController::class, 'list'])->name('kelolaPengguna.list');
        // Route::get('/create', [UserController::class, 'create'])->name('kelolaPengguna.create');
        // Route::post('/', [UserController::class, 'store'])->name('kelolaPengguna.store');
        Route::get('/{id}', [UserController::class, 'show'])->name('kelolaPengguna.show');
        Route::get('/{id}/edit', [UserController::class, 'edit'])->name('kelolaPengguna.edit');
        Route::put('/{id}', [UserController::class, 'update'])->name('kelolaPengguna.update');
        // Route::delete('/{id}', [UserController::class, 'destroy'])->name('kelolaPengguna.destroy');
    });
});
