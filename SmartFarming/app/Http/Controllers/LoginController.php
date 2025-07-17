<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Session;
use RealRashid\SweetAlert\Facades\Alert;

class LoginController extends Controller
{
    public function index()
    {
        return view('login.index');
    }

    public function authenticate(Request $request)
    {
        $request->validate([
            'email' => 'required|email',
            'password' => 'required',
        ]);

        $remember = $request->has('remember');

        $response = Http::timeout(300)->withBasicAuth($request->email, $request->password)
            ->post('http://labai.polinema.ac.id:3042/auth/login', [
                'remember_me' => $remember
            ]);

        if ($response->successful()) {
            $data = $response->json();

            // Ambil token dan jwtToken
            $token = $data['data']['token'] ?? null;
            $jwt   = $data['data']['jwtToken'] ?? null;

            if ($token && $jwt) {
                // Decode JWT
                $decoded = $this->decodeJwtPayload($jwt);

                $isBanned = $decoded['user']['is_ban'] ?? false;
                if ($isBanned) {
                    Alert::toast('Akun Anda diblokir, silakan hubungi admin.', 'error');
                    return back()->withInput();
                }

                // Simpan ke session
                session([
                    'token' => $token,
                    'jwt'   => $jwt,
                    'user_id'  => $decoded['user']['id'] ?? 'guest',
                    'username'  => $decoded['user']['username'] ?? 'guest',
                    'nama'  => $decoded['user']['fullname'] ?? 'Guest',
                    'email' => $decoded['user']['email'] ?? null,
                    'created_at' => $decoded['user']['created_at'] ?? null,
                    'updated_at' => $decoded['user']['updated_at'] ?? null,
                    'role_name'  => $decoded['user']['role']['name'] ?? 'GUEST',
                    'role_code'  => $decoded['user']['role']['code'] ?? 'Guest',
                ]);
                Alert::toast('Selamat Datang, ' . session('nama'), 'success');
                return redirect('/dashboard');
            } else {
                Alert::toast('Respon login tidak valid', 'error');
                return back()->withInput();
            }
        } else {
            $message = $response->json()['message'] ?? 'Login gagal';
            Alert::toast($message, 'error');
            return back()->withInput();
        }
    }

    function decodeJwtPayload($token)
    {
        $parts = explode('.', $token);
        if (count($parts) !== 3) return null;

        $payload = base64_decode(strtr($parts[1], '-_', '+/'));
        return json_decode($payload, true);
    }

    public function logout(Request $request)
    {
        $token = session('token');

        if ($token) {
            $response = Http::timeout(300)->withToken($token)
                ->post('http://labai.polinema.ac.id:3042/auth/logout');

            if ($response->successful()) {
                Session::flush();
                Alert::toast('Berhasil logout!', 'success');
            } else {
                Alert::toast('Gagal menghubungi server logout.', 'error');
            }
        }
        return redirect()->route('login.index');
    }
}
