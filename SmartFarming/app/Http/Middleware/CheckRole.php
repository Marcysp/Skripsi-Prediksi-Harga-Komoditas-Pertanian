<?php

namespace App\Http\Middleware;

use Closure;
use Illuminate\Http\Request;
use Symfony\Component\HttpFoundation\Response;

class CheckRole
{
    /**
     * Handle an incoming request.
     *
     * @param  \Closure(\Illuminate\Http\Request): (\Symfony\Component\HttpFoundation\Response)  $next
     */
    public function handle(Request $request, Closure $next, ...$roles): Response
    {
        $userRole = session('role_code'); // Ambil role dari session (gunakan 'role_code' sesuai yang kamu simpan)

        if (!in_array($userRole, $roles)) {
            return redirect()->route('login.index')->with('error', 'Akses ditolak');
        }

        return $next($request);
    }
}
