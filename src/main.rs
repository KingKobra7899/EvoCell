use minifb::{Key, Window, WindowOptions, ScaleMode, Scale};
use raqote::{DrawTarget, SolidSource, Source, PathBuilder, DrawOptions, Point};
use std::f32::consts::PI;

fn render_circle(radius: f32, position: (f32, f32), dt: &mut DrawTarget) {
    let mut pb = PathBuilder::new();
    pb.arc(position.0, position.1, radius, 0.0, 2.0 * PI); // radians
    let path = pb.finish();

    dt.fill(
        &path,
        &Source::Solid(SolidSource::from_unpremultiplied_argb(0xff, 0xff, 0x80, 0x80)),
        &DrawOptions::new(),
    );
}

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

    while window.is_open() && !window.is_key_down(Key::Escape) {
        dt.clear(SolidSource::from_unpremultiplied_argb(0xff, 0x20, 0x20, 0x30));

        render_circle(50.0, (250.0, 250.0), &mut dt);

        window.update_with_buffer(dt.get_data(), WIDTH, HEIGHT).unwrap();
    }
}
