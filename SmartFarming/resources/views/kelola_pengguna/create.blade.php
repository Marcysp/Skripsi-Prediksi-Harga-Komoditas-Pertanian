@extends('layout.template')
@section('title', 'Kelola Pengguna | Agrilink Vocpro')
@section('content')
    <div class="container-fluid">
        <div class="card">
            <div class="card-body">
                <form method="POST" action="{{ route('kelolaPengguna.store') }}" class="form-horizontal"
                    enctype="multipart/form-data" id="tambahPengguna">
                    @csrf
                    <div class=" form-group row">
                        <div class="col-md-6 mt-3">
                            <div class="form-group">
                                <label for="username" class="form-label">Username</label>
                                <input type="text" class="form-control @error('username') is-invalid @enderror"
                                    id="username" name="username" placeholder="Masukkan Username Pengguna"
                                    value="{{ old('username') }}" required autofocus>
                                <p><small class="text-muted">Wajib Diisi!</small></p>
                                @if ($errors->has('username'))
                                    <span class="text-danger">{{ $errors->first('username') }}</span>
                                @endif
                            </div>
                            <div class="form-group">
                                <label for="password" class="form-label">Password</label>
                                <div class="input-group">
                                    <input type="password" class="form-control @error('password') is-invalid @enderror"
                                        id="password" name="password" placeholder="Masukkan Password Pengguna" required>
                                    <div class="input-group-append">
                                        <span class="input-group-text" id="toggle-password"
                                            style="cursor: pointer; padding: 0.7rem 0.6rem;">
                                            <i class="fa fa-eye" id="eye-icon"></i>
                                        </span>
                                    </div>
                                </div>
                                <p><small class="text-muted">Wajib Diisi!</small></p>
                                @if ($errors->has('password'))
                                    <span class="text-danger">{{ $errors->first('password') }}</span>
                                @endif
                            </div>
                            <div class="form-group">
                                <label for="role" class="form-label">Role</label>
                                <div class="form-group">
                                    <select class="choices form-select @error('urole_id') is-invalid @enderror"
                                        name="urole_id" id="urole_id" required>
                                        <option value="">- Pilih Role -</option>
                                        @foreach ($role as $item)
                                            <option value="{{ $item->role_id }}">{{ $item->role_name }}</option>
                                        @endforeach
                                    </select>
                                    <p><small class="text-muted">Wajib Diisi!</small></p>
                                </div>
                                @if ($errors->has('urole_id'))
                                    <span class="text-danger">{{ $errors->first('urole_id') }}</span>
                                @endif
                            </div>
                            <div class="form-group">
                                <label for="fullname" class="form-label">Nama</label>
                                <input type="text" class="form-control @error('nama') is-invalid @enderror"
                                    id="nama" name="nama" placeholder="Masukkan Nama Pengguna"
                                    value="{{ old('nama') }}" required>
                                <p><small class="text-muted">Wajib Diisi!</small></p>
                                @if ($errors->has('nama'))
                                    <span class="text-danger">{{ $errors->first('nama') }}</span>
                                @endif
                            </div>
                            <div class="form-group">
                                <label for="no_hp" class="form-label">Nomor Handphone</label>
                                <input type="text" class="form-control @error('no_hp') is-invalid @enderror"
                                    id="no_hp" name="no_hp" placeholder="Masukkan No Hp Pengguna"
                                    value="{{ old('no_hp') }}">
                                <p><small class="text-muted">Boleh Dikosongi.</small></p>
                                @if ($errors->has('no_hp'))
                                    <span class="text-danger">{{ $errors->first('no_hp') }}</span>
                                @endif
                            </div>
                            <div class="form-group">
                                <label for="alamat" class="form-label">Alamat</label>
                                <textarea class="form-control @error('alamat') is-invalid @enderror" id="alamat" name="alamat" rows="3"
                                    placeholder="Masukkan Alamat Pengguna">{{ old('alamat') }}</textarea>
                                <p><small class="text-muted">Boleh Dikosongi.</small></p>
                                @if ($errors->has('alamat'))
                                    <span class="text-danger">{{ $errors->first('alamat') }}</span>
                                @endif
                            </div>
                            <div class="form-group">
                                <label for="gaji_pokok" class="form-label">Gaji Pokok</label>
                                <input type="number" class="form-control @error('gaji_pokok') is-invalid @enderror"
                                    id="gaji_pokok" name="gaji_pokok" placeholder="Masukkan Gaji Pokok Pengguna"
                                    value="{{ old('gaji_pokok') }}" required>
                                <p><small class="text-muted">Wajib Diisi!</small></p>
                                @if ($errors->has('gaji_pokok'))
                                    <span class="text-danger">{{ $errors->first('gaji_pokok') }}</span>
                                @endif
                            </div>
                            <div class="form-group">
                                <label for="komisi" class="form-label">Komisi</label>
                                <input type="number" class="form-control @error('komisi') is-invalid @enderror"
                                    id="komisi" name="komisi" placeholder="Masukkan Komisi Pengguna"
                                    value="{{ old('komisi') }}">
                                <p><small class="text-muted">Boleh Dikosongi.</small></p>
                                @if ($errors->has('komisi'))
                                    <span class="text-danger">{{ $errors->first('komisi') }}</span>
                                @endif
                            </div>
                            <div class="form-group">
                                <label for="tunjangan" class="form-label">Tunjangan</label>
                                <input type="number" class="form-control @error('tunjangan') is-invalid @enderror"
                                    id="tunjangan" name="tunjangan" placeholder="Masukkan Tunjangan Pengguna"
                                    value="{{ old('tunjangan') }}">
                                <p><small class="text-muted">Boleh Dikosongi.</small></p>
                                @if ($errors->has('tunjangan'))
                                    <span class="text-danger">{{ $errors->first('tunjangan') }}</span>
                                @endif
                            </div>
                            <div class="form-group">
                                <label for="potongan_gaji" class="form-label">Potongan Gaji</label>
                                <input type="number" class="form-control @error('potongan_gaji') is-invalid @enderror"
                                    id="potongan_gaji" name="potongan_gaji" placeholder="Masukkan Potongan Gaji Pengguna"
                                    value="{{ old('potongan_gaji') }}">
                                <p><small class="text-muted">Boleh Dikosongi.</small></p>
                                @if ($errors->has('potongan_gaji'))
                                    <span class="text-danger">{{ $errors->first('potongan_gaji') }}</span>
                                @endif
                            </div>
                            <div class="form-group">
                                <label for="posisi" class="form-label">Posisi</label>
                                <div class="form-group">
                                    <select class="choices form-select @error('posisi') is-invalid @enderror"
                                        name="posisi" id="posisi" required>
                                        <option value="">- Pilih Posisi -</option>
                                        <option value="Manager">Manager</option>
                                        <option value="Teknisi">Teknisi</option>
                                        <option value="Analis Tambak">Analis Tambak</option>
                                        <option value="Pemilik Tambak">Pemilik Tambak</option>
                                    </select>
                                    <p><small class="text-muted">Wajib Diisi!</small></p>
                                </div>
                                @if ($errors->has('posisi'))
                                    <span class="text-danger">{{ $errors->first('posisi') }}</span>
                                @endif
                            </div>
                        </div>
                        <div class="col-md-6 d-flex justify-content-center align-items-center mt-3">
                            <div class="form-group">
                                <div class="col">
                                    <div class="row mb-3">
                                        <div class="drop-zone"
                                            style="width: 300px; height: 300px; border: 2px dashed #ccc; border-radius: 5px; display: flex; justify-content: center; align-items: center; cursor: pointer;">
                                            <div class="text-center">
                                                <i class="fa-solid fa-cloud-arrow-up"
                                                    style="height: 50px; font-size: 50px"></i>
                                                <p class="mt-2">Seret lalu letakkan file di sini</p>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="row mb-3">
                                        <span class="text-center">Atau</span>
                                    </div>
                                    <div class="row mb-5">
                                        <div class="form-file">
                                            <label class="form-file-label" for="foto">
                                                <span class="form-file-text">Choose file...</span>
                                                <span class="form-file-button">Browse</span>
                                                <input type="file" class="drop-zone__input" id="foto"
                                                    name="foto">
                                            </label>
                                        </div>
                                    </div>
                                    @if ($errors->has('foto'))
                                        <div class="row alert alert-danger">
                                            <span class="text-center">{{ $errors->first('foto') }}</span>
                                        </div>
                                    @endif
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="d-flex justify-content-between">
                        <button type="button" class="btn btn-sm btn-danger"
                            onclick="window.location.href='{{ url('kelolaPengguna') }}'"
                            style="background-color: #DC3545; border-color: #DC3545" id="btn-kembali">Kembali</button>
                        <button type="submit" class="btn btn-primary btn-sm"
                            style="background-color: #007BFF; border-color: #007BFF" id="btn-simpan">Simpan</button>
                    </div>
                </form>
            </div>
        </div>
    </div>
@endsection
@push('js')
    <script>
        document.getElementById('toggle-password').addEventListener('click', function(e) {
            const passwordInput = document.getElementById('password');
            const eyeIcon = document.getElementById('eye-icon');

            if (passwordInput.type === 'password') {
                passwordInput.type = 'text';
                eyeIcon.classList.remove('fa-eye');
                eyeIcon.classList.add('fa-eye-slash');
            } else {
                passwordInput.type = 'password';
                eyeIcon.classList.remove('fa-eye-slash');
                eyeIcon.classList.add('fa-eye');
            }
        });
    </script>
@endpush
