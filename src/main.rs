use minifb::{Key, Window, WindowOptions, ScaleMode, Scale};
use nalgebra::Vector2;
use raqote::{DrawTarget, SolidSource};
use std::time::Instant;

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

    physics_solver.add_particle_grid(10, 10, Vector2::new(200.0, 200.0), 5.0, 10.0, 1.0);

    // FPS tracking variables
    let mut frame_count = 0;
    let mut last_fps_time = Instant::now();
    let mut last_frame_time = Instant::now();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let frame_start = Instant::now();
        
        dt.clear(SolidSource::from_unpremultiplied_argb(0xff, 0x20, 0x20, 0x30));
        physics_solver.update(&mut dt, 0.03, 1, Vector2::new(0.0, 10.0));
        window.update_with_buffer(dt.get_data(), WIDTH, HEIGHT).unwrap();
        
        frame_count += 1;
        
        
        let now = Instant::now();
        if now.duration_since(last_fps_time).as_secs() >= 1 {
            let fps = frame_count as f64 / now.duration_since(last_fps_time).as_secs_f64();
            println!("FPS: {:.1}", fps);
            frame_count = 0;
            last_fps_time = now;
        }
        
        last_frame_time = frame_start;
    }
}