@extends('layout.template')
@section('title', 'Edit Pengguna')
@section('content')
    <div class="card">
        <div class="card-body">
            @empty($user)
                <div class="alert alert-danger alert-dismissible">
                    <h5><i class="icon fas fa-ban white"></i> Kesalahan!</h5> Data yang Anda cari tidak ditemukan.
                </div>
            @else
                <form method="POST" action="{{ route('kelolaPengguna.update', $user->account_id) }}" class="form-horizontal"
                    enctype="multipart/form-data" id="editPengguna">
                    @csrf {!! method_field('PUT') !!}
                    <div class=" form-group row">
                        <div class="col-12 col-md-12">
                            <div class="form-group">
                                <label for="username" class="form-label">Username</label>
                                <input type="text" class="form-control @error('username') is-invalid @enderror"
                                    id="username" name="username" value="{{ old('username', $user->username) }}"
                                    placeholder="Masukkan Username Pengguna" required autofocus>
                                <p><small class="text-muted">Wajib Diisi!</small></p>
                                @if ($errors->has('username'))
                                    <span class="text-danger">{{ $errors->first('username') }}</span>
                                @endif
                            </div>
                            <div class="form-group">
                                <label for="email" class="form-label">Email</label>
                                <input type="text" class="form-control @error('email') is-invalid @enderror" id="email"
                                    name="email" value="{{ old('email', $user->email) }}"
                                    placeholder="Masukkan Email Pengguna" required>
                                <p><small class="text-muted">Wajib Diisi!</small></p>
                                @if ($errors->has('email'))
                                    <span class="text-danger">{{ $errors->first('email') }}</span>
                                @endif
                            </div>
                            <div class="form-group">
                                <label for="password" class="form-label">Password</label>
                                <div class="input-group">
                                    <input type="password" class="form-control @error('password') is-invalid @enderror"
                                        id="password" name="password"
                                        placeholder="Abaikan (Jangan Diisi) Jika Tidak Ingin Mengganti Password">
                                    <span class="input-group-text" id="toggle-password" style="cursor: pointer;">
                                        <i class="fa fa-eye" id="eye-icon"></i>
                                    </span>
                                </div>
                                <p><small class="text-muted">Boleh Dikosongi.</small></p>
                                @if ($errors->has('password'))
                                    <span class="text-danger">{{ $errors->first('password') }}</span>
                                @endif
                            </div>
                            <div class="form-group">
                                <label for="role" class="form-label">Role</label>
                                <div class="form-group">
                                    <select class="choices form-select @error('urole_id') is-invalid @enderror" name="urole_id"
                                        id="urole_id" required>
                                        <option value="">- Pilih Role -</option>
                                        @foreach ($roles as $item)
                                            <option value="{{ $item->role_id }}"
                                                {{ $item->role_id == $user->urole_id ? 'selected' : '' }}>
                                                {{ $item->role_name }}
                                            </option>
                                        @endforeach
                                    </select>
                                    <p><small class="text-muted">Wajib Diisi!</small></p>
                                </div>
                                @if ($errors->has('urole_id'))
                                    <span class="text-danger">{{ $errors->first('urole_id') }}</span>
                                @endif
                            </div>
                            <div class="form-group">
                                <label for="no_hp" class="form-label">Nama Lengkap</label>
                                <input type="text" class="form-control @error('fullname') is-invalid @enderror"
                                    id="fullname" name="fullname" value="{{ old('fullname', $user->fullname) }}"
                                    placeholder="Masukkan Nama Lengkap">
                                <p><small class="text-muted">Wajib Diisi.</small></p>
                                @if ($errors->has('fullname'))
                                    <span class="text-danger">{{ $errors->first('fullname') }}</span>
                                @endif
                            </div>
                            <div class="form-group">
                                <label for="is_ban" class="form-label">Status Akun</label>
                                <div class="form-group">
                                    <select class="choices form-select @error('is_ban') is-invalid @enderror" name="is_ban"
                                        id="is_ban" required>
                                        <option value="">- Status Akun -</option>
                                        <option value="0" {{ $user->is_ban == 0 ? 'selected' : '' }}>Aktif</option>
                                        <option value="1" {{ $user->is_ban == 1 ? 'selected' : '' }}>Banned</option>
                                    </select>
                                    <p><small class="text-muted">Wajib Diisi!</small></p>
                                </div>
                                @if ($errors->has('urole_id'))
                                    <span class="text-danger">{{ $errors->first('urole_id') }}</span>
                                @endif
                            </div>
                        </div>
                    </div>
                    <div class="d-flex justify-content-between">
                        <button type="button" class="btn btn-sm btn-danger"
                            onclick="window.location.href='{{ route('kelolaPengguna.index') }}'"
                            style="background-color: #DC3545; border-color: #DC3545" id="btn-kembali">Kembali</button>
                        <button type="submit" class="btn btn-primary btn-sm"
                            style="background-color: #227066; border-color: #227066" id="btn-simpan">Simpan</button>
                    </div>
                </form>
            @endempty
        </div>
    </div>
@endsection
@push('css')
@endpush
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
