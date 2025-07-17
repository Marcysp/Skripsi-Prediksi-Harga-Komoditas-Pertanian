<!DOCTYPE html>
<html lang="en">

<head>
    <title>@yield('title', 'Agrilink Vocpro')</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=0, minimal-ui">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="description" content="Web Monitoring dan Visualisasi Data Smart Farming">
    <meta name="keywords" content="Web Monitoring Smart Farming">
    <meta name="author" content="Rasyid">
    <meta name="csrf-token" content="{{ csrf_token() }}">
    <link rel="icon" href="{{ asset('storage/asset_web/LOGO NO BG.png') }}" type="image/x-icon">
    <link rel="stylesheet"
        href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@300;400;500;600;700&display=swap"
        id="main-font-link">
    <link rel="stylesheet" href="{{ asset('Mantis-Bootstrap-1.0.0/dist/assets/fonts/tabler-icons.min.css') }}">
    <link rel="stylesheet" href="{{ asset('Mantis-Bootstrap-1.0.0/dist/assets/fonts/feather.css') }}">
    <link rel="stylesheet" href="{{ asset('Mantis-Bootstrap-1.0.0/dist/assets/fonts/fontawesome.css') }}">
    <link rel="stylesheet" href="{{ asset('Mantis-Bootstrap-1.0.0/dist/assets/fonts/material.css') }}">
    <link rel="stylesheet" href="{{ asset('Mantis-Bootstrap-1.0.0/dist/assets/css/style.css') }}" id="main-style-link">
    <link rel="stylesheet" href="{{ asset('Mantis-Bootstrap-1.0.0/dist/assets/css/style-preset.css') }}">
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css" rel="stylesheet">
    <link rel="stylesheet" href="{{ asset('css/app.css') }}">
    <link rel="stylesheet" href="{{ asset('css/riwayatDataDHT.css') }}">
    <link rel="stylesheet" href="{{ asset('css/riwayatDataNPK.css') }}">
    <link rel="stylesheet" href="{{ asset('css/riwayatDataCuaca.css') }}">
    <link rel="stylesheet" href="{{ asset('css/kelolaPengguna.css') }}">
    @stack('css')
</head>

<body data-pc-preset="preset-1" data-pc-direction="ltr" data-pc-theme="light">
    @include('sweetalert::alert')
    <div class="loader-bg">
        <div class="loader-track">
            <div class="loader-fill"></div>
        </div>
    </div>
    @include('layout.sidebar')
    @include('layout.header')
    <div class="pc-container">
        <div class="pc-content">
            @include('layout.breadcrumb')
            <section class="content">
                @yield('content')
            </section>
        </div>
    </div>
    @include('layout.footer')
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>
    <link href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/momentjs/latest/moment.min.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/daterangepicker/daterangepicker.css" />
    <script src="https://cdn.jsdelivr.net/npm/daterangepicker/daterangepicker.min.js"></script>
    <script src="{{ asset('Mantis-Bootstrap-1.0.0/dist/assets/js/plugins/popper.min.js') }}"></script>
    <script src="{{ asset('Mantis-Bootstrap-1.0.0/dist/assets/js/plugins/simplebar.min.js') }}"></script>
    <script src="{{ asset('Mantis-Bootstrap-1.0.0/dist/assets/js/plugins/bootstrap.min.js') }}"></script>
    <script src="{{ asset('Mantis-Bootstrap-1.0.0/dist/assets/js/fonts/custom-font.js') }}"></script>
    <script src="{{ asset('Mantis-Bootstrap-1.0.0/dist/assets/js/pcoded.js') }}"></script>
    <script src="{{ asset('Mantis-Bootstrap-1.0.0/dist/assets/js/plugins/feather.min.js') }}"></script>
    <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
    <script>
        layout_change('light');
    </script>
    <script>
        change_box_container('false');
    </script>
    <script>
        layout_rtl_change('false');
    </script>
    <script>
        preset_change("preset-1");
    </script>
    <script>
        font_change("Public-Sans");
    </script>
    <script>
        document.addEventListener("DOMContentLoaded", function() {
            const links = document.querySelectorAll(".pc-link");

            function setActiveLink() {
                const currentUrl = window.location.href;
                links.forEach(link => {
                    if (link.href === currentUrl) {
                        link.classList.add("active");
                    } else {
                        link.classList.remove("active");
                    }
                });
            }
            setActiveLink();
            links.forEach(link => {
                link.addEventListener("click", function() {
                    links.forEach(l => l.classList.remove("active"));
                    this.classList.add("active");
                });
            });
        });
    </script>
    <script>
        $.ajaxSetup({
            headers: {
                'X-CSRF-TOKEN': $('meta[name="csrf-token"]').attr('content')
            }
        });
    </script>
    @stack('js')
</body>

</html>
