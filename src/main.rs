use minifb::{Key, Window, WindowOptions, ScaleMode, Scale};
use nalgebra::Vector2;
use raqote::{DrawTarget, SolidSource};
mod particle;
use particle::Particle;

fn main() {
    const WIDTH: usize = 500;
    const HEIGHT: usize = 500;

    let mut window = Window::new(
        "window",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: false,
            scale_mode: ScaleMode::Stretch,
            scale: Scale::X1,
            ..WindowOptions::default()
        },
    ).unwrap();

    let mut dt = DrawTarget::new(WIDTH as i32, HEIGHT as i32);
    let mut p: Particle  = Particle::new(Vector2::new(250.0, 250.0), 1.5, 20.1);
    while window.is_open() && !window.is_key_down(Key::Escape) {
        dt.clear(SolidSource::from_unpremultiplied_argb(0xff, 0x20, 0x20, 0x30));
        p.integrate_pos(0.01, Vector2::new(0.0, 10.0));
        p.render(&mut dt);
        window.update_with_buffer(dt.get_data(), WIDTH, HEIGHT).unwrap();
    }
}
