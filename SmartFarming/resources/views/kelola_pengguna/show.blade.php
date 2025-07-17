@extends('layout.template')
@section('title', 'Kelola Pengguna | Agrilink Vocpro')
@section('content')
    <div class="container-fluid">
        @empty($user)
            <div class="alert alert-danger alert-dismissible">
                <h5><i class="icon fas fa-ban"></i> Kesalahan!</h5> Data yang Anda cari tidak ditemukan.
            </div>
        @else
            <div class="main-body">
                <div class="row gutters-sm">
                    <div class="col-12 col-lg-12">
                        <div class="card mb-3">
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-sm-2">
                                        <h6 class="mb-0">Username</h6>
                                    </div>
                                    <div class="col-sm-10 text-secondary">
                                        : {{ $user->username ?? '-' }}
                                    </div>
                                </div>
                                <hr>
                                <div class="row">
                                    <div class="col-sm-2">
                                        <h6 class="mb-0">Password</h6>
                                    </div>
                                    <div class="col-sm-10 text-secondary">
                                        : ********
                                    </div>
                                </div>
                                <hr>
                                <div class="row">
                                    <div class="col-sm-2">
                                        <h6 class="mb-0">Nama Lengkap</h6>
                                    </div>
                                    <div class="col-sm-10 text-secondary">
                                        : {{ $user->fullname }}
                                    </div>
                                </div>
                                <hr>
                                <div class="row">
                                    <div class="col-sm-2">
                                        <h6 class="mb-0">Email</h6>
                                    </div>
                                    <div class="col-sm-10 text-secondary">
                                        : {{ $user->email }}
                                    </div>
                                </div>
                                <hr>
                                <div class="row">
                                    <div class="col-sm-2">
                                        <h6 class="mb-0">Role</h6>
                                    </div>
                                    <div class="col-sm-10 text-secondary">
                                        : {{ $user->role_name }}
                                    </div>
                                </div>
                                <hr>
                                <div class="row">
                                    <div class="col-sm-2">
                                        <h6 class="mb-0">Status Akun</h6>
                                    </div>
                                    <div class="col-sm-10 text-secondary">
                                        : @if ($user->is_ban)
                                            <span class="badge bg-danger">Banned</span>
                                        @else
                                            <span class="badge bg-success">Aktif</span>
                                        @endif
                                    </div>
                                </div>
                                <hr>
                                <div class="row">
                                    <div class="col-sm-2">
                                        <h6 class="mb-0">Akun Dibuat</h6>
                                    </div>
                                    <div class="col-sm-10 text-secondary">
                                        : {{ \Carbon\Carbon::parse($user->created_at)->translatedFormat('d F Y, H:i') }} WIB
                                    </div>
                                </div>
                                <hr>
                                <div class="row">
                                    <div class="col-sm-2">
                                        <h6 class="mb-0">Akun Diperbarui</h6>
                                    </div>
                                    <div class="col-sm-10 text-secondary">
                                        : {{ \Carbon\Carbon::parse($user->updated_at)->translatedFormat('d F Y, H:i') }} WIB
                                    </div>
                                </div>
                                <hr>
                                <div class="row">
                                    <div class="col-sm-2">
                                        <h6 class="mb-0">Akun Dihapus</h6>
                                    </div>
                                    <div class="col-sm-10 text-secondary">
                                        :
                                        {{ $user->deleted_at ? \Carbon\Carbon::parse($user->created_at)->translatedFormat('d F Y, H:i') . ' WIB' : 'Data Tidak Dihapus' }}
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="d-flex justify-content-end me-3">
                            <a href="{{ route('kelolaPengguna.index') }}" class="btn btn-sm btn-primary my-3"
                                style="background-color: #227066; border-color: #227066;">Kembali</a>
                        </div>
                    </div>
                </div>
            </div>
        @endempty
    </div>
@endsection
@push('js')
@endpush
