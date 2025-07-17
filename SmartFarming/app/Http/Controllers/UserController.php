<?php

namespace App\Http\Controllers;

use App\Models\AccountModel;
use App\Models\RoleModel;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Hash;
use Yajra\DataTables\Facades\DataTables;
use Illuminate\Support\Str;
use RealRashid\SweetAlert\Facades\Alert;

class UserController extends Controller
{
    public function index()
    {
        $breadcrumb = (object) [
            'title' => 'Kelola Pengguna',
            'paragraph' => 'Kelola semua akun pengguna dengan mudah dan efisien. Atur peran, ubah informasi, dan pastikan data tetap terkini.',
            'list' => [
                ['label' => 'Kelola Pengguna', 'url' => route('kelolaPengguna.index')],
                ['label' => 'List'],
            ]
        ];

        $roles = RoleModel::select([
            'user.role.*'
        ])->get();

        $activeMenu = 'kelolaPengguna';
        return view('kelola_pengguna.index', compact(
            'breadcrumb',
            'activeMenu',
            'roles',
        ));
    }

    public function list(Request $request)
    {
        $user = AccountModel::select([
            'user.account.id as account_id',
            'user.account.username',
            'user.account.fullname',
            'user.account.email',
            'user.account.avatar',
            'user.account.is_ban',
            'user.account.urole_id',
            'user.account.created_at',
            'user.account.updated_at',
            'user.account.deleted_at',
            'user.role.code as role_code',
            'user.role.name as role_name'
        ])->leftJoin('user.role', 'user.account.urole_id', '=', 'user.role.id');

        if ($request->role) {
            $user->whereHas('role', function ($query) use ($request) {
                $query->where('code', $request->role); // sesuaikan jika role disimpan di kolom lain seperti 'name'
            });
        }

        if ($request->filled('is_ban') && in_array($request->is_ban, ['0', '1'])) {
            $user->where('user.account.is_ban', $request->is_ban);
        }


        return DataTables::of($user)
            ->addIndexColumn()

            // ->editColumn('username', fn($row) => $row->username ?? '-')
            ->editColumn('email', fn($row) => $row->email ?? '-')
            ->editColumn('fullname', fn($row) => $row->fullname ?? '-')

            // ->editColumn(
            //     'avatar',
            //     fn($row) =>
            //     $row->avatar
            //         ? '<span class="badge bg-success">Tersedia</span>'
            //         : '<span class="badge bg-secondary">Tidak Ada</span>'
            // )

            ->editColumn(
                'is_ban',
                fn($row) =>
                $row->is_ban
                    ? '<span class="badge bg-danger">Banned</span>'
                    : '<span class="badge bg-success">Aktif</span>'
            )

            ->addColumn('role_name', fn($row) => optional($row->role)->name ?? '-')

            // ✅ Kolom Aksi
            ->addColumn('aksi', function ($row) {
                $btn = '<div class="text-center d-flex gap-1 justify-content-center">';
                $btn .= '<a href="' . route('kelolaPengguna.show', $row->account_id) . '" class="btn btn-sm btn-info">Show</a>';
                $btn .= '<a href="' . route('kelolaPengguna.edit', $row->account_id) . '" class="btn btn-sm btn-warning">Edit</a>';
                $btn .= '<button class="btn btn-sm btn-danger" onclick="hapusPengguna(\'' . $row->id . '\')">Hapus</button>';
                $btn .= '</div>';
                return $btn;
            })

            ->rawColumns(['avatar', 'is_ban', 'aksi']) // biar HTML badge dan tombol ditampilkan
            ->make(true);
    }

    // public function create()
    // {
    //     $breadcrumb = (object) [
    //         'title' => 'Tambah Pengguna',
    //         'paragraph' => 'Masukkan data pengguna baru dengan lengkap untuk proses registrasi akun.',
    //         'list' => [
    //             ['label' => 'Kelola Pengguna', 'url' => route('kelolaPengguna.index')],
    //             ['label' => 'Tambah Pengguna'],
    //         ]
    //     ];

    //     $roles = RoleModel::select([
    //         'user.role.id as role_id',
    //         'user.role.code as role_code',
    //         'user.role.name as role_name',
    //         'user.role.created_at as role_createdAt',
    //         'user.role.updated_at as role_updatedAt',
    //     ])->get();

    //     $activeMenu = 'kelolaPengguna';
    //     return view('kelola_pengguna.create', compact(
    //         'breadcrumb',
    //         'activeMenu',
    //         'roles',
    //     ));
    // }

    public function show($id)
    {
        if (!Str::isUuid($id)) {
            abort(404, 'ID tidak valid');
        }

        $breadcrumb = (object) [
            'title' => 'Detail Pengguna',
            'paragraph' => 'Lihat informasi lengkap akun pengguna.',
            'list' => [
                ['label' => 'Kelola Pengguna', 'url' => route('kelolaPengguna.index')],
                ['label' => 'Detail Pengguna'],
            ]
        ];

        $activeMenu = 'kelolaPengguna';

        $user = AccountModel::select([
            'user.account.id as account_id',
            'user.account.username',
            'user.account.fullname',
            'user.account.email',
            'user.account.avatar',
            'user.account.is_ban',
            'user.account.urole_id',
            'user.account.created_at',
            'user.account.updated_at',
            'user.account.deleted_at',
            'user.role.name as role_name'
        ])->leftJoin('user.role', 'user.account.urole_id', '=', 'user.role.id')
            ->where('user.account.id', $id)
            ->first();

        return view('kelola_pengguna.show', compact(
            'breadcrumb',
            'activeMenu',
            'user',
        ));
    }

    public function edit($id)
    {
        if (!Str::isUuid($id)) {
            abort(404, 'ID tidak valid');
        }

        $breadcrumb = (object) [
            'title' => 'Edit Pengguna',
            'paragraph' => 'Perbarui informasi akun pengguna untuk memastikan data tetap akurat dan terkini.',
            'list' => [
                ['label' => 'Kelola Pengguna', 'url' => route('kelolaPengguna.index')],
                ['label' => 'List'],
            ]
        ];
        $activeMenu = 'kelolaPengguna';

        $user = AccountModel::select([
            'user.account.id as account_id',
            'user.account.username',
            'user.account.fullname',
            'user.account.email',
            'user.account.avatar',
            'user.account.is_ban',
            'user.account.urole_id',
            'user.account.created_at',
            'user.account.updated_at',
            'user.account.deleted_at',
            'user.role.name as role_name'
        ])->leftJoin('user.role', 'user.account.urole_id', '=', 'user.role.id')
            ->where('user.account.id', $id)
            ->first();

        $roles = RoleModel::select([
            'user.role.id as role_id',
            'user.role.code as role_code',
            'user.role.name as role_name',
            'user.role.created_at as role_createdAt',
            'user.role.updated_at as role_updatedAt',
        ])->get();

        return view('kelola_pengguna.edit', compact(
            'breadcrumb',
            'activeMenu',
            'user',
            'roles'
        ));
    }

    public function update(Request $request, $id)
    {
        $user = AccountModel::select([
            'user.account.id as account_id',
            'user.account.username',
            'user.account.fullname',
            'user.account.email',
            'user.account.avatar',
            'user.account.is_ban',
            'user.account.urole_id',
            'user.account.created_at',
            'user.account.updated_at',
            'user.account.deleted_at',
            'user.role.name as role_name'
        ])->leftJoin('user.role', 'user.account.urole_id', '=', 'user.role.id')
            ->where('user.account.id', $id)
            ->first();

        $validated = $request->validate([
            'fullname' => 'required|string|max:255',
            'username' => 'required|string|max:50|unique:user.account,username,' . $id,
            'email' => 'required|email|unique:user.account,email,' . $id,
            'password' => 'nullable|min:6',
            'urole_id' => 'required|exists:user.role,id',
            'is_ban' => 'required|boolean',
        ]);

        if ($request->filled('password')) {
            $validated['password'] = Hash::make($request->password);
        } else {
            unset($validated['password']); // jangan ubah password jika kosong
        }

        $user->update($validated);
        Alert::toast('Data pengguna berhasil diperbarui!', 'success');
        return redirect()->route('kelolaPengguna.index');
    }
}
