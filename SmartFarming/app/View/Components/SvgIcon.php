<?php

namespace App\View\Components;

use Illuminate\View\Component;

class SvgIcon extends Component
{
    public $icon;

    /**
     * Create a new component instance.
     *
     * @param string $icon Nama file SVG tanpa ekstensi
     * @return void
     */
    public function __construct($icon)
    {
        $this->icon = $icon;
    }

    /**
     * Get the view / contents that represent the component.
     *
     * @return \Illuminate\View\View|string
     */
    public function render()
    {
        return view('components.svg-icon');
    }
}
