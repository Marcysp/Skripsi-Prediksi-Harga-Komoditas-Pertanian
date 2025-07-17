<?php

namespace App\Http\Middleware;

use Closure;
use Illuminate\Http\Request;
use Symfony\Component\HttpFoundation\Response;

class CustomAuthMiddleware
{
    /**
     * Handle an incoming request.
     *
     * @param  \Closure(\Illuminate\Http\Request): (\Symfony\Component\HttpFoundation\Response)  $next
     */
    public function handle(Request $request, Closure $next): Response
    {
        // Cek apakah token login tersimpan di session
        if (!session()->has('token')) {
            return redirect()->route('login.index')->with('error', 'Silakan login terlebih dahulu.');
        }
        return $next($request);
    }
}
