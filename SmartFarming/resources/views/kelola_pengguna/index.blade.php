@extends('layout.template')
@section('title', 'Kelola Pengguna | Agrilink Vocpro')
@section('content')
    <div class="container-fluid">
        <div class="d-flex row justify-content-between mb-2">
            <div class="form-group col-12 col-lg-3">
                <label for="start_date" class="form-label">Filter Status Akun:</label>
                <select id="is_ban" class="form-select">
                    <option value="">- Semua Status -</option>
                    <option value="0">Aktif</option>
                    <option value="1">Banned</option>
                </select>
            </div>
            <div class="form-group col-12 col-lg-3">
                <label for="start_date" class="form-label">Filter Role:</label>
                <select class="form-select" name="role" id="role">
                    <option value="">- Semua Role -</option>
                    @foreach ($roles as $role)
                        <option value="{{ $role->code }}">{{ $role->name }}</option>
                    @endforeach
                </select>
            </div>
        </div>
        <div class="card">
            <div class="card-header">
                <div class="d-flex justify-content-end">
                    <a class="btn btn-sm btn-primary" href="#"
                        style="background-color: #227066; border-color: #227066;">Tambah</a>
                </div>
            </div>
            <div class="card-body">
                <div class="responsive-table-wrapper">
                    <div class="responsive-table-wrapper">
                        <table class="table align-middle table-striped table-bordered" id="table_kelola_pengguna">
                            <thead>
                                <tr class="text-center">
                                    <th>No</th>
                                    {{-- <th>Username</th> --}}
                                    <th>Nama Lengkap</th>
                                    <th>Email</th>
                                    <th>Role</th>
                                    {{-- <th>Avatar</th> --}}
                                    <th>Status Akun</th>
                                    {{-- <th>Terdaftar</th> --}}
                                    <th>Aksi</th>
                                </tr>
                            </thead>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
@endsection
@push('js')
    <script>
        $(document).ready(function() {
            // Ambil nilai filter dari localStorage (jika ada)
            if (localStorage.getItem('filter_role')) {
                $('#role').val(localStorage.getItem('filter_role'));
            }
            if (localStorage.getItem('filter_is_ban')) {
                $('#is_ban').val(localStorage.getItem('filter_is_ban'));
            }

            table = $('#table_kelola_pengguna').DataTable({
                processing: true,
                serverSide: true,
                ajax: {
                    url: '{{ route('kelolaPengguna.list') }}',
                    type: 'POST',
                    data: function(d) {
                        _token: '{{ csrf_token() }}' // wajib untuk POST
                        d.role = $('#role').val();
                        d.is_ban = $('#is_ban').val();
                    }
                },
                columns: [{
                        data: 'DT_RowIndex',
                        className: 'text-center',
                        orderable: false,
                        searchable: false
                    },
                    // {
                    //     data: 'username',
                    //     className: "text-center",
                    //     orderable: false,
                    //     searchable: false
                    // },
                    {
                        data: 'fullname',
                        className: "",
                        orderable: true,
                        searchable: true
                    },
                    {
                        data: 'email',
                        className: "text-center",
                        orderable: false,
                        searchable: false
                    },
                    {
                        data: 'role_name',
                        className: "text-center",
                        orderable: false,
                        searchable: false
                    },
                    // {
                    //     data: 'avatar',
                    //     className: "text-center",
                    //     orderable: false,
                    //     searchable: false
                    // },
                    {
                        data: 'is_ban',
                        className: "text-center",
                        orderable: false,
                        searchable: false
                    },
                    // {
                    //     data: 'created_at',
                    //     className: "text-center",
                    //     orderable: false,
                    //     searchable: false
                    // },
                    {
                        data: 'aksi',
                        className: "text-center",
                        orderable: false,
                        searchable: false
                    },
                ]
            });
        });

        $('#role, #is_ban').on('change', function() {
            // Simpan ke localStorage
            localStorage.setItem('filter_role', $('#role').val());
            localStorage.setItem('filter_is_ban', $('#is_ban').val());

            // Reload table
            table.ajax.reload();
        });


        // Fungsi hapus
        function hapusPengguna(id) {
            if (confirm("Yakin ingin menghapus pengguna ini?")) {
                $.ajax({
                    url: '/admin/user/' + id,
                    type: 'DELETE',
                    data: {
                        _token: '{{ csrf_token() }}'
                    },
                    success: function(response) {
                        $('#table_kelola_pengguna').DataTable().ajax.reload();
                        alert('Pengguna berhasil dihapus.');
                    },
                    error: function() {
                        alert('Terjadi kesalahan saat menghapus.');
                    }
                });
            }
        }
    </script>
@endpush
