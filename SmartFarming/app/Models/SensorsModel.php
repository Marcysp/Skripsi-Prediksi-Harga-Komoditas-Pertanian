<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class SensorsModel extends Model
{
    use HasFactory;

    protected $table = 'sensors';
    protected $primaryKey = 'id';
    protected $fillable = ['name', 'public_name', 'desc', 'created_at', 'updated_at', 'deleted_at', 'bed_location_id'];

    public function bed_location()
    {
        return $this->belongsTo(BedLocationModel::class, 'bed_location_id', 'id');
    }

    public function sensor_readings()
    {
        return $this->hasMany(SensorReadingsModel::class, 'sensor_id', 'id');
    }
}
