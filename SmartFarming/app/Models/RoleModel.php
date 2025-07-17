<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\HasMany;

class RoleModel extends Model
{
    use HasFactory;

    protected $table = 'user.role';
    protected $primaryKey = 'id';
    protected $fillable = ['code', 'name', 'created_at', 'updated_at'];

    public function account(): HasMany
    {
        return $this->hasMany(AccountModel::class, 'urole_id', 'id');
    }
}
