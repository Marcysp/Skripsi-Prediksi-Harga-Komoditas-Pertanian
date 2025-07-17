<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class CuacaModel extends Model
{
    use HasFactory;

    protected $table = 'weather_data';
    protected $primaryKey = 'id';
    protected $fillable = ['time', 'temperature_2m', 'weather_code', 'cloud_cover', 'wind_speed_10m'];
}
