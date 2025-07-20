use minifb::{Key, Window, WindowOptions, ScaleMode, Scale};
use nalgebra::Vector2;
use raqote::{DrawTarget, SolidSource};

mod solver;
use solver::PhysicsSolver;
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
    let mut physics_solver = PhysicsSolver::new(WIDTH as i32, HEIGHT as i32);

    physics_solver.add_particle(Vector2::new(100.0, 250.0), 1.0, 12.5);
    physics_solver.add_particle(Vector2::new(225.0, 250.0), 1.0, 12.5);
    physics_solver.add_particle(Vector2::new(300.0, 250.0), 1.0, 12.5);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        dt.clear(SolidSource::from_unpremultiplied_argb(0xff, 0x20, 0x20, 0x30));
        physics_solver.update(&mut dt, 0.01, 1, Vector2::new(0.0, 10.0));
        window.update_with_buffer(dt.get_data(), WIDTH, HEIGHT).unwrap();
    }
}