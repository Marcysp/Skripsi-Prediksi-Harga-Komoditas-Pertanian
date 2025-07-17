<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class ApiTokenModel extends Model
{
    use HasFactory;

    protected $table = 'user.api_tokens';
    protected $primaryKey = 'id';
    protected $fillable = ['user_id', 'name', 'type', 'token', 'created_at', 'updated_at'];

    public function account(): BelongsTo
    {
        return $this->belongsTo(AccountModel::class, 'user_id', 'id');
    }
}
