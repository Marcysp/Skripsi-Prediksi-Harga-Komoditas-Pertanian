<div class="container">
    <div class="page-header py-3">
        <div class="page-block">
            <div class="row align-items-center">
                <div class="col-12 col-lg-6">
                    <div class="page-header-title">
                        <h3 class="mb-1">{{ $breadcrumb->title }}</h3>
                        <p class="text-muted">{{ $breadcrumb->paragraph }}</p>
                    </div>
                </div>
                <div class="col-12 col-lg-6 text-md-end">
                    <ul class="breadcrumb mb-0 justify-content-md-end">
                        @foreach ($breadcrumb->list as $key => $value)
                            @if ($key == count($breadcrumb->list) - 1)
                                <li class="breadcrumb-item active">{{ $value['label'] }}</li>
                            @else
                                <li class="breadcrumb-item hover-underline"><a
                                        href="{{ $value['url'] }}">{{ $value['label'] }}</a></li>
                            @endif
                        @endforeach
                    </ul>
                </div>
            </div>
        </div>
    </div>
</div>
