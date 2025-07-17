<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;

class AccountModel extends Model
{
    use HasFactory;

    protected $table = 'user.account';
    protected $primaryKey = 'id';
    protected $fillable = ['urole_id', 'username', 'password', 'email', 'google_id', 'fullname', 'avatar', 'is_ban', 'created_at', 'updated_at', 'deleted_at'];

    public function role(): BelongsTo
    {
        return $this->belongsTo(RoleModel::class, 'urole_id', 'id');
    }

    public function apiToken(): HasMany
    {
        return $this->hasMany(ApiTokenModel::class, 'user_id', 'id');
    }
}
