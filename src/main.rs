use nalgebra::Vector2;
use std::time::Instant;
use winit::{
    event::{Event, KeyEvent, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

mod solver; // Assuming solver.rs exists and contains PhysicsSolver
use solver::PhysicsSolver;

fn main() {
    // Define window dimensions
    const WIDTH: usize = 1000;
    const HEIGHT: usize = 1000;

    // Initialize the winit event loop. This is the core of event handling.
    let event_loop = EventLoop::new().expect("Failed to create EventLoop");

    // Create a new window for our application.
    let window_attributes = Window::default_attributes()
        .with_title("WGPU Physics Simulation") // Set the window's title bar text
        .with_inner_size(winit::dpi::PhysicalSize::new(WIDTH as u32, HEIGHT as u32)); // Set the window's resolution
    
    let window = event_loop.create_window(window_attributes)
        .expect("Failed to create window"); // Handle any errors during window creation

    // Initialize the physics solver with the window's dimensions.
    let mut physics_solver = PhysicsSolver::new(WIDTH as i32, HEIGHT as i32);

    // Add a grid of particles to the simulation for visual demonstration.
    physics_solver.add_particle_grid(10, 10, Vector2::new(200.0, 200.0), 5.0, 10.0, 1.0);

    // Variables for tracking performance (FPS) and frame timing.
    let mut frame_count = 0;
    let mut last_fps_time = Instant::now();
    let mut last_frame_time = Instant::now();

    // Start the winit event loop. This loop will continuously process events
    // and drive our application's updates and rendering.
    event_loop.run(move |event, active_event_loop| {
        // Set the control flow to 'Poll' to ensure the application continuously
        // processes events and requests redraws, suitable for a game loop.
        active_event_loop.set_control_flow(ControlFlow::Poll);

        // Match on the type of event received.
        match event {
            // Handle events specific to our window.
            Event::WindowEvent {
                event, window_id,
            } if window_id == window.id() => {
                match event {
                    // If the user requests to close the window (e.g., by clicking the 'X' button).
                    WindowEvent::CloseRequested => {
                        println!("Close requested. Exiting.");
                        active_event_loop.exit(); // Signal the event loop to terminate.
                    }
                    // Handle keyboard input events.
                    WindowEvent::KeyboardInput {
                        event: KeyEvent {
                            physical_key: PhysicalKey::Code(KeyCode::Escape), // Check if the Escape key was pressed.
                            state,
                            ..
                        },
                        ..
                    } => {
                        // Only exit if the Escape key was pressed down, not released.
                        if state.is_pressed() {
                            println!("Escape key pressed. Exiting.");
                            active_event_loop.exit(); // Signal the event loop to terminate.
                        }
                    }
                    // This event is triggered when the window needs to be redrawn.
                    // This is where our main game loop logic (physics update, rendering) goes.
                    WindowEvent::RedrawRequested => {
                        // Record the start time of the current frame for performance tracking.
                        let frame_start = Instant::now();

                        // --- Physics Update Logic ---
                        // Update the physics simulation. We're assuming `PhysicsSolver::update`
                        // no longer needs a `dt` (draw target) argument as WGPU will handle rendering.
                        physics_solver.update(0.03, 1, Vector2::new(0.0, 10.0));
                        
                        // --- WGPU Rendering Placeholder ---
                        // This is where you would integrate your `wgpu` rendering code.
                        // In a typical WGPU application, you would:
                        // 1. Get the current texture from the window's swap chain.
                        // 2. Create a command encoder.
                        // 3. Begin a render pass, setting up the output target (the texture).
                        // 4. Issue draw commands using your WGPU pipeline and buffers (likely derived from `physics_solver` data).
                        // 5. End the render pass and submit the command buffer to the GPU queue.
                        // 6. Present the rendered texture to the screen.
                        // Example: Your custom `render(&window, &physics_solver)` function would be called here.

                        // --- FPS Calculation ---
                        frame_count += 1; // Increment the frame counter.

                        let now = Instant::now();
                        // Check if one second has passed since the last FPS measurement.
                        if now.duration_since(last_fps_time).as_secs() >= 1 {
                            // Calculate FPS and print it to the console.
                            let fps = frame_count as f64 / now.duration_since(last_fps_time).as_secs_f64();
                            println!("FPS: {:.1}", fps);
                            frame_count = 0; // Reset frame count.
                            last_fps_time = now; // Update the last FPS measurement time.
                        }

                        last_frame_time = frame_start; // Store the start time for the next frame's delta calculation (if needed).
                    }
                    _ => (), // Ignore other window events.
                }
            }
            // This event is triggered when the event loop is about to go idle
            // and wait for new events. It's the perfect place to do continuous
            // updates and then request a redraw.
            Event::AboutToWait => {
                window.request_redraw(); // Ask the window to redraw itself.
            }
            _ => (), // Ignore any other events.
        }
    }).expect("EventLoop run failed"); // Handle any errors that cause the event loop to fail.
}