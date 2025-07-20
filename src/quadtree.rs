use nalgebra::Vector2;

pub struct Rect{
    x:f32,
    y:f32,
    w:f32,
    h:f32
}

impl Rect{
    pub fn new(x:f32, y:f32, w:f32, h:f32) -> Rect{
        return Rect{x:x, y:y, w:w, h:h};
    }

    pub fn point_is_in(&self, point: Vector2<f32>) -> bool{

    }

    pub fn intersects(&self, othert: Rect) -> bool{

    }
}