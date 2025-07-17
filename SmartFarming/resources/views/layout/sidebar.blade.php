<nav class="pc-sidebar">
    <div class="navbar-wrapper">
        <div class="m-header">
            <a href="{{ route('dashboard.index') }}">
                <img src="{{ Storage::url('asset_web/logo.png') }}" class="img-fluid p-3" alt="logo">
            </a>
        </div>
        <div class="navbar-content">
            <ul class="pc-navbar">
                <li class="pc-item">
                    <a href="{{ url('/dashboard') }}" class="pc-link {{ $activeMenu == 'dashboard' ? 'active' : '' }}">
                        <span class="pc-micon"><i class="ti ti-dashboard"></i></span>
                        <span class="pc-mtext">Dasbor</span>
                    </a>
                </li>
                <li class="pc-item pc-caption">
                    <label>Sensor</label>
                    <i class="ti ti-dashboard"></i>
                </li>
                <li class="pc-item pc-hasmenu">
                    <a href="#" class="pc-link">
                        <span class="pc-micon">
                            <i class="ti ti-access-point"></i>
                        </span>
                        <span class="pc-mtext">Pemantauan Sensor</span>
                        <span class="pc-arrow">
                            <i data-feather="chevron-right"></i>
                        </span>
                    </a>
                    <ul class="pc-submenu">
                        <li class="pc-item">
                            <a class="pc-link {{ $activeMenu == 'monitoringSensorNPK' ? 'active' : '' }}"
                                href="{{ route('monitoringSensorNPK.index') }}">Sensor NPK</a>
                        </li>
                        <li class="pc-item">
                            <a class="pc-link {{ $activeMenu == 'monitoringSensorDHT' ? 'active' : '' }}"
                                href="{{ route('monitoringSensorDHT.index') }}">Sensor DHT</a>
                        </li>
                    </ul>
                </li>
                <li class="pc-item pc-hasmenu">
                    <a href="#" class="pc-link">
                        <span class="pc-micon">
                            <i class="ti ti-history"></i>
                        </span>
                        <span class="pc-mtext">Riwayat Sensor</span>
                        <span class="pc-arrow">
                            <i data-feather="chevron-right"></i>
                        </span>
                    </a>
                    <ul class="pc-submenu">
                        <li class="pc-item">
                            <a class="pc-link {{ $activeMenu == 'riwayatDataNPK' ? 'active' : '' }}"
                                href="{{ route('riwayatDataNPK.index') }}">Sensor NPK</a>
                        </li>
                        <li class="pc-item">
                            <a class="pc-link {{ $activeMenu == 'riwayatDataDHT' ? 'active' : '' }}"
                                href="{{ route('riwayatDataDHT.index') }}">Sensor DHT</a>
                        </li>
                    </ul>
                </li>
                <li class="pc-item pc-caption">
                    <label>Cuaca</label>
                    <i class="ti ti-dashboard"></i>
                </li>
                <li class="pc-item pc-hasmenu">
                    <a href="#" class="pc-link">
                        <span class="pc-micon">
                            <i class="ti ti-cloud"></i>
                        </span>
                        <span class="pc-mtext">Pemantauan Cuaca</span>
                        <span class="pc-arrow">
                            <i data-feather="chevron-right"></i>
                        </span>
                    </a>
                    <ul class="pc-submenu">
                        <li class="pc-item">
                            <a class="pc-link {{ $activeMenu == 'monitoringCuaca' ? 'active' : '' }}"
                                href="{{ route('monitoringCuaca.index') }}">Prediksi Cuaca</a>
                        </li>
                        <li class="pc-item">
                            <a class="pc-link {{ $activeMenu == 'cuacaTerkini' ? 'active' : '' }}"
                                href="{{ route('cuacaTerkini.index') }}">Cuaca Terkini</a>
                        </li>
                    </ul>
                </li>
                <li class="pc-item">
                    <a href="{{ route('riwayatDataCuaca.index') }}"
                        class="pc-link {{ $activeMenu == 'riwayatDataCuaca' ? 'active' : '' }}">
                        <span class="pc-micon">
                            <i class="ti ti-history"></i>
                        </span>
                        <span class="pc-mtext">Riwayat Prediksi Cuaca</span>
                    </a>
                </li>
                <li class="pc-item pc-caption">
                    <label>Komoditas</label>
                    <i class="ti ti-dashboard"></i>
                </li>
                <li class="pc-item pc-hasmenu">
                    <a href="#" class="pc-link">
                        <span class="pc-micon">
                            <i class="bi bi-bar-chart-line"></i>
                        </span>
                        <span class="pc-mtext">Harga Komoditas</span>
                        <span class="pc-arrow">
                            <i data-feather="chevron-right"></i>
                        </span>
                    </a>
                    <ul class="pc-submenu">
                        <li class="pc-item">
                            <a class="pc-link {{ $activeMenu == 'prediksiHarga' ? 'active' : '' }}"
                                href="{{ route('komoditas.prediksi') }}">Prediksi Harga</a>
                        </li>
                        <li class="pc-item">
                            <a class="pc-link {{ $activeMenu == 'trendHarga' ? 'active' : '' }}"
                                href="{{ route('komoditas.trend') }}">Trend Harga</a>
                        </li>
                        <li class="pc-item">
                            <a class="pc-link {{ $activeMenu == 'riwayatHarga' ? 'active' : '' }}"
                                href="{{ route('komoditas.riwayat') }}">Riwayat Harga</a>
                        </li>
                    </ul>
                </li>
                @if (session('role_code') === 'ADMN')
                    <li class="pc-item pc-caption">
                        <label>Manajemen</label>
                        <i class="ti ti-dashboard"></i>
                    </li>
                    <li class="pc-item">
                        <a href="{{ route('kelolaPengguna.index') }}"
                            class="pc-link {{ $activeMenu == 'kelolaPengguna' ? 'active' : '' }}">
                            <span class="pc-micon">
                                <i class="ti ti-user-check"></i>
                            </span>
                            <span class="pc-mtext">Kelola Pengguna</span>
                        </a>
                    </li>
                @endif
            </ul>
        </div>
    </div>
</nav>
