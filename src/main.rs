// main.rs
use nalgebra::Vector2;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::{KeyEvent, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

mod solver;
mod gpu_renderer; // Import our new module

const WIDTH: usize = 1000;
const HEIGHT: usize = 1000;

struct App {
    window: Option<Window>,
    gpu_renderer: Option<gpu_renderer::GpuRenderer>,
    physics_solver: solver::PhysicsSolver,
    frame_count: u32,
    last_fps_time: Instant,
}

impl App {
    fn new() -> Self {
        let mut physics_solver = solver::PhysicsSolver::new(WIDTH as i32, HEIGHT as i32);
        
        
        physics_solver.add_particle_grid(40, 50, Vector2::new(200.0, 100.0), 7.0, 14.0, 1.0);

        Self {
            window: None,
            gpu_renderer: None,
            physics_solver,
            frame_count: 0,
            last_fps_time: Instant::now(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_none() {
            let window_attributes = Window::default_attributes()
                .with_title("WGPU Physics Simulation (Per-Pixel)")
                .with_inner_size(winit::dpi::PhysicalSize::new(WIDTH as u32, HEIGHT as u32));

            let window = event_loop
                .create_window(window_attributes)
                .expect("Failed to create window");

            // Initialize GpuRenderer
            self.gpu_renderer = Some(pollster::block_on(gpu_renderer::GpuRenderer::new(&window)));
            self.window = Some(window);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // Poll regularly to keep the simulation and rendering active
        event_loop.set_control_flow(ControlFlow::Poll);

        match event {
            WindowEvent::CloseRequested => {
                println!("Close requested. Exiting.");
                event_loop.exit();
            }

            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key: PhysicalKey::Code(KeyCode::Escape),
                    state,
                    ..
                },
                ..
            } => {
                if state.is_pressed() {
                    println!("Escape key pressed. Exiting.");
                    event_loop.exit();
                }
            }

            WindowEvent::RedrawRequested => {
                // Update physics
                self.physics_solver.update(0.03, 1, Vector2::new(0.0, 10.0)); // Gravity: (0, 10)

                // Prepare particle data for GPU
                let num_physics_particles = self.physics_solver.positions.len();
                let mut gpu_particles: Vec<gpu_renderer::GpuParticle> = Vec::with_capacity(num_physics_particles);
                for i in 0..num_physics_particles {
                    gpu_particles.push(gpu_renderer::GpuParticle {
                        position: [self.physics_solver.positions[i].x, self.physics_solver.positions[i].y],
                        radius: self.physics_solver.radii[i],
                        _padding: 0.0,
                    });
                }

                // Render the current state
                if let (Some(renderer), Some(window)) = (&mut self.gpu_renderer, &self.window) {
                    renderer.render(window, &gpu_particles, num_physics_particles as u32);
                }

                // FPS Calculation
                self.frame_count += 1;
                let now = Instant::now();
                if now.duration_since(self.last_fps_time).as_secs() >= 1 {
                    let fps = self.frame_count as f64 / now.duration_since(self.last_fps_time).as_secs_f64();
                    println!("FPS: {:.1}", fps);
                    self.frame_count = 0;
                    self.last_fps_time = now;
                }
            }
            // Handle window resizing to update surface configuration
            WindowEvent::Resized(physical_size) => {
                if let Some(renderer) = &mut self.gpu_renderer {
                    renderer.resize(physical_size);
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        // Request a redraw every frame to keep the simulation moving
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("Failed to create EventLoop");
    let mut app = App::new();

    event_loop.run_app(&mut app).expect("EventLoop run failed");
}