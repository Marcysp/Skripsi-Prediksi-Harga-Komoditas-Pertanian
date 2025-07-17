<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class SensorReadingsModel extends Model
{
    use HasFactory;
    protected $table = 'sensor_readings';

    protected $primaryKey = 'id';
    public $incrementing = true;
    protected $keyType = 'int';

    protected $fillable = [
        'sensor_id',
        'payload',
        'created_at',
        'read_at',
    ];

    protected $casts = [
        'payload' => 'array', // otomatis decode jsonb ke array
        'created_at' => 'datetime',
        'read_at' => 'datetime',
    ];

    public function sensor()
    {
        return $this->belongsTo(SensorsModel::class, 'sensor_id', 'id');
    }
}
