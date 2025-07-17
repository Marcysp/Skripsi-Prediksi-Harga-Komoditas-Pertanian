<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

class ProfilController extends Controller
{
    public function index()
    {
        $breadcrumb = (object) [
            'title' => 'User Profile',
            'paragraph' => 'Selamat datang di halaman profil. Berikut informasi akun Anda yang terdaftar.',
            'list' => [
                ['label' => 'Profil', 'url' => route('profil.index')],
                ['label' => session('nama')],
            ]
        ];
        $activeMenu = 'profil';
        $user = session('user');
        return view('profil.index', compact(
            'breadcrumb',
            'activeMenu',
            'user',
        ));
    }

    public function edit()
    {
        $user = [
            'user_id' => session('user_id'),
            'username' => session('username'),
            'nama' => session('nama'),
            'email' => session('email'),
            'created_at' => session('created_at'),
            'updated_at' => session('updated_at'),
            'role_name' => session('role_name'),
            'role_code' => session('role_code'),
        ];

        $breadcrumb = (object) [
            'title' => 'Ubah Profil',
            'paragraph' => 'Selamat datang di halaman ubah profil. Berikut informasi akun Anda yang terdaftar.',
            'list' => [
                ['label' => 'Profil', 'url' => route('profil.index')],
                ['label' => session('nama')],
            ]
        ];
        $activeMenu = 'editProfil';
        return view('profil.edit', compact(
            'breadcrumb',
            'activeMenu',
            'user',
        ));
    }
}
