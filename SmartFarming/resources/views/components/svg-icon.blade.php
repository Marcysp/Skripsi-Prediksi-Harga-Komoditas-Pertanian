@php
    $svgPath = resource_path("svg/{$icon}.svg");
    if (file_exists($svgPath)) {
        echo file_get_contents($svgPath);
    } else {
        echo "<!-- SVG file not found: {$icon}.svg -->";
    }
@endphp
